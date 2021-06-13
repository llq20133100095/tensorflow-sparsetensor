# tensorflow-sparsetensor

# 1.背景
最近在做模型训练，发现在导入大量数据时，由于要进行预处理（concat和reshape操作等），导致内存会占满，使得程序出错。由于输入数据存在大量的稀疏情况，想着能不能输入数据时利用稀疏矩阵进行保存，然后输入到模型中进行训练。

# 2.稀疏矩阵输入构造
python中scipy.sparse模块，能够有效的对输入数据进行稀疏化存储。但缺点在于稀疏矩阵必定只有两维的操作，但一般图片分类设置到多个维度，因此需要提前把输入数据reshape成两维矩阵。

假设现有的图片大小为`[3, 31,31]`。其中$31*31$是图片的大小，而$3$是图片channel。有20w的数据，则内存需要提前存储`[200000, 3, 31, 31]`的容量大小，这对于小内存的机器来说是不可行的。因此需要先把这20w图片进行稀疏化矩阵操作：

- 首先需要循环读取每一张图片，同时进行稀疏化操作
```python
import numpy as np
from scipy.sparse import csr_matrix

input_data = []
with open(file, "r") as f:
	while True:
		line = f.readline()
		if line:
			fig = csr_matrix(np.reshape(line, [3, 31*31]))
			input_data.append(fig)
		else:
			break
```

- 然后对取得的list进行稀疏化矩阵拼接，会得到一个`[200000 * 3, 31*31]`的稀疏矩阵，这样就能够有效的在内存中进行存储
```python
from scipy import sparse
input_data = sparse.vstack(input_data)
```

# 3.稀疏数据模型训练
## 3.1 利用tensorflow中的tf.SparseTensor
在tensorflow2.0中，可以包装对应的稀疏矩阵进行输入。

- 首先把scipy的稀疏矩阵，转换成tf.SparseTensor格式
```python
def get_sparse_tensor(input_data):
    indices = list(zip(*input_data.nonzero()))
    return tf.SparseTensor(indices=indices, values=np.float32(input_data.data), dense_shape=input_data.get_shape())

```

- 然后在模型构建时，需要把输入的数据进行reshape，重新转换成`[batch_size, 3, 31, 31]`，这样才能用卷积的方法进行训练，核心代码如下：
```python
# 把稀疏矩阵进行reshape操作
input_global_map = tf.sparse.reshape(input_global_map, [-1, 3, 31, 31])
# 把sparsetensor转换成普通tensor，这样模型才能够训练
input_global_map = tf.sparse.to_dense(input_global_map)
```