import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ReLU


# class StgcnLayer(nn.Module):
class StgcnLayer(Model):
    """[Training] Applies a spatial temporal graph convolution over an input graph sequence.

    Processes the entire video capture during training; it is mandatory to retain intermediate values
    for backpropagation (hence no FIFOs allowed in training). Results of training with either layer
    are identical, it is simply a nuissance of autodiff frameworks.
    All arguments are positional to enforce separation of concern and pass the responsibility for
    model configuration up in the chain to the envoking program (config file).

    TODO:
        ``1.`` validate documentation.

    Shape:
        - Input[0]:     :math:`(N, C_{in}, L, V)` - Input graph frame.
        - Input[1]:     :math:`(P, V, V)` - Graph adjacency matrix.
        - Output[0]:    :math:`(N, C_{out}, L, V)` - Output graph frame.

        where
            :math:`N` is the batch size.

            :math:`C_{in}` is the number of input channels (features).

            :math:`C_{out}` is the number of output channels (features).

            :math:`L` is the video capture length.

            :math:`V` is the number of graph nodes.

            :math:`P` is the number of graph partitions.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            num_joints,
            stride,
            num_partitions,
            dropout,
            residual):
        """
        Args:
            in_channels : ``int``
                Number of input sample channels/features.

            out_channels : ``int``
                Number of channels produced by the convolution.

            kernel_size : ``int``
                Size of the temporal window Gamma.

            num_joints : ``int``
                Number of joint nodes in the graph.

            stride : ``int``
                Stride of the temporal reduction.

            num_partitions : ``int``
                Number of partitions in selected strategy.
                Must correspond to the first dimension of the adjacency tensor.

            dropout : ``float``
                Dropout rate of the final output.

            residual : ``bool``
                If ``True``, applies a residual connection.
        """

        super().__init__()

        # temporal kernel Gamma is symmetric (odd number)
        # assert len(kernel_size) == 1
        assert kernel_size % 2 == 1

        self.num_partitions = num_partitions
        self.num_joints = num_joints
        self.stride = stride
        self.kernel_size = kernel_size

        self.out_channels = out_channels

        # convolution of incoming frame
        # (out_channels is a multiple of the partition number
        # to avoid for-looping over several partitions)
        # partition-wise convolution results are basically stacked across channel-dimension

##需要修改 Need to add the Input Layer
##需要修改 Need to change the parameters (in_channels) for ()? and also the Input layer.
        # self.conv = nn.Conv2d(in_channels, out_channels * num_partitions, kernel_size=1, bias=False)
        self.conv = Conv2D(out_channels * num_partitions, kernel_size=1, use_bias=False)

        # normalization and dropout on main branch
##需要修改 Need to add Out_channels Parameter ('num_features') in the BatchNormalization()?
        # self.bn_relu = nn.Sequential(
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU())
        self.bn_relu = Sequential([
            BatchNormalization(),
            ReLU()])


        # residual branch
        if not residual:
            self.residual = lambda _: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
##需要修改 Need to add Out_channels Parameter ('num_features') in the BatchNormalization()?
            # self.residual = nn.Sequential(
            #     nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            #     nn.BatchNorm2d(out_channels))
            self.residual = Sequential([
                Conv2D(out_channels, kernel_size=1, bias=False),
                BatchNormalization()])

        # activation of branch sum
        # if no resnet connection, prevent ReLU from being applied twice
        if not residual:
            self.do = Dropout(dropout)
        else:
            self.do = Sequential([
                ReLU(),
                Dropout(dropout)])


    def forward(self, x, A):
        # TODO: replace with unfold -> fold calls
        # lower triangle matrix for temporal accumulation that mimics FIFO behavior
        # capture_length = x.size(2)
        # device = torch.device("cuda:{0}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
        # lt_matrix = torch.zeros(capture_length, capture_length, device=device)
        # for i in range(self.kernel_size // self.stride):
        #     lt_matrix += F.pad(
        #         torch.eye(
        #             capture_length - self.stride * i,
        #             device=device),
        #         (i * self.stride, 0, 0, i * self.stride))
##需要检查 Is the dataType set correctly?
        capture_length = x.size(2)
        device = tf.device("GPU" if tf.test.is_gpu_available() else "CPU")
        lt_matrix = tf.zeros((capture_length, capture_length), dtype=tf.float32)
        for i in range(self.kernel_size // self.stride):
            lt_matrix += tf.pad(
                tf.eye(
                    capture_length - self.stride * i,
                    dtype=tf.float32),
                (i * self.stride, 0, 0, i * self.stride))
        # must register matrix as a buffer to automatically move to GPU with model.to_device()
        # for PyTorch v1.0.1
        # self.register_buffer('lt_matrix', lt_matrix)

        # residual branch
        res = self.residual(x)

        # spatial convolution of incoming frame (node-wise)
        x = self.conv(x)

        # convert to the expected dimension order and add the partition dimension
        # reshape the tensor for multiplication with the adjacency matrix
        # (convolution output contains all partitions, stacked across the channel dimension)
        # split into separate 4D tensors, each corresponding to a separate partition
##需要检查 Is it axis parameter correct? Axis=3 or Axis=1?
        # x = torch.split(x, self.out_channels, dim=1)
        x = tf.split(x, self.out_channels, axis=1)

        # concatenate these 4D tensors across the partition dimension
##需要检查 Is it axis parameter correct?
        # x = torch.stack(x, -1)
        x = tf.stack(x, -1)

        # change the dimension order for the correct broadcating of the adjacency matrix
        # (N,C,L,V,P) -> (N,L,P,C,V)
        # x = x.permute(0, 2, 4, 1, 3)
        x = tf.transpose(x,perm=[0, 2, 4, 1, 3])

        # single multiplication with the adjacency matrices (spatial selective addition, across partitions)
        # x = torch.matmul(x, A)
        x = tf.matmul(x, A)

        # sum temporally by multiplying features with the Toeplitz matrix
        # reorder dimensions for correct broadcasted multiplication (N,L,P,C,V) -> (N,P,C,V,L)
        # x = x.permute(0, 2, 3, 4, 1)
        x = tf.transpose(x,perm=[0, 2, 3, 4, 1])
        # x = torch.matmul(x, lt_matrix)
        x = tf.matmul(x, lt_matrix)

        # sum across partitions (N,C,V,L)
        # x = torch.sum(x, dim=(1))
        x = tf.reduce_sum(x, axis=1)
        # match the dimension ordering of the input (N,C,V,L) -> (N,C,L,V)
        # x = x.permute(0, 1, 3, 2)
        x = tf.transpose(x, perm=[0, 1, 3, 2])

        # normalize the output of the st-gcn operation and activate
        x = self.bn_relu(x)

        # add the branches (main + residual), activate and dropout
        return self.do(x + res)

