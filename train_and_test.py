import pandas as pd
import paddle
import numpy as np

%pylab inline
import seaborn as sns

train_df = pd.read_csv('data/data137267/train.csv.zip')
test_df = pd.read_csv('data/data137267/test.csv.zip')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#pca=PCA(200)
#train_values=pca.fit_transform(train_df.values[:, :-1])
#test_values=pca.fit_transform(test_df.values)
scaler = StandardScaler()
scaler.fit(train_df.values[:, :-1])
train_df.iloc[:, :-1] = scaler.transform(train_df.values[:, :-1])
test_df.iloc[:, :] = scaler.transform(test_df.values)

class GaussianNoise(paddle.nn.Layer):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            #print(type(din))
            #print(din.shape)
            return din + paddle.randn(din.shape) * self.stddev
        return din


class Classifier(paddle.nn.Layer):
    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Classifier, self).__init__()
        
        self.inconv= paddle.nn.Sequential(
            paddle.nn.Conv1D(1, 16, kernel_size=7, stride=2, padding=3),
            paddle.nn.BatchNorm1D(16),
            paddle.nn.ReLU(),
        )
        self.conv1 = paddle.nn.Conv1D(in_channels=16, out_channels=16, kernel_size=1,stride=1,padding=1)
        self.conv2 = paddle.nn.Conv1D(in_channels=16, out_channels=16, kernel_size=3,stride=1,padding=1)
        self.conv3 = paddle.nn.Conv1D(in_channels=16, out_channels=64, kernel_size=1,stride=1)
        self.shortcut = paddle.nn.Conv1D(in_channels=16, out_channels=64, kernel_size=1,stride=1,padding=1)
        self.flatten = paddle.nn.Flatten()
        self.dropout = paddle.nn.Dropout()
        self.fc = paddle.nn.Linear(in_features=1472, out_features=6)
        self.relu = paddle.nn.ReLU()
        self.pool =  paddle.nn.MaxPool1D(kernel_size=3, stride=2, padding=1)
        self.pool2=  paddle.nn.AvgPool1D(6)
        self.softmax = paddle.nn.Softmax()
        self.noise= GaussianNoise(0.005)
        self.bn1=paddle.nn.BatchNorm1D(16)
        self.bn2=paddle.nn.BatchNorm1D(32)
        self.bn3=paddle.nn.BatchNorm1D(64)

    # 网络的前向计算,创新的内容为加入batch—normanization,resnet
    def forward(self, inputs):
        #print(inputs.shape)
        
        x = self.noise(inputs)       ##这里是创新的内容，加入噪声
        x = self.inconv(x)
        #print(x.shape)
        x=self.pool(x)
        h=x
        x = self.relu(self.conv1(x))
        x=self.bn1(x)
        #h  = x #设置shortcut
        x = self.relu(self.conv2(x))
        x=self.bn1(x)
        #print(x.shape)
        x = self.dropout(x)
        #print(x.shape)
        x = self.relu(self.conv3(x))
        #print(x.shape)
        x=self.bn3(x)
        #print(x.shape)
        
        h=self.shortcut(h)
        #print(h.shape)
        x=x+h
        #print(x.shape)
        x=self.pool2(x)
        x = self.flatten(x)
        #print(x.shape)
        x = self.fc(x)
        x = self.softmax(x)

        return x
      
      
      
      model = Classifier()
      model.train()
      opt = paddle.optimizer.Adam(learning_rate=0.005, parameters=model.parameters())
      loss_fn = paddle.nn.CrossEntropyLoss()
      
      EPOCH_NUM = 200   # 设置外层循环次数,基于baseline，让train轮次变多一点
      BATCH_SIZE = 16  # 设置batch大小
      training_data = train_df.iloc[:-1000].values.astype(np.float32)
      val_data = train_df.iloc[-1000:].values.astype(np.float32)

      training_data = training_data.reshape(-1, 1, 562)
      val_data = val_data.reshape(-1, 1, 562)
      
      # 定义外层循环
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)
    
    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
    
    # 定义内层循环
    for iter_id, mini_batch in enumerate(mini_batches):
        model.train()
        x = np.array(mini_batch[:,:, :-1]) # 获得当前批次训练数据
        y = np.array(mini_batch[:,:, -1:]) # 获得当前批次训练标签
        
        # 将numpy数据转为飞桨动态图tensor的格式
        features = paddle.to_tensor(x)
        y = paddle.to_tensor(y)
        
        # 前向计算
        predicts = model(features)
        
        # 计算损失
        loss = loss_fn(predicts, y.flatten().astype(int))
        avg_loss = paddle.mean(loss)

        # 反向传播，计算每层参数的梯度值
        avg_loss.backward()
        # 更新参数，根据设置好的学习率迭代一步
        opt.step()
        # 清空梯度变量，以备下一轮计算
        opt.clear_grad()

        # 训练与验证
        if iter_id%2000==0 and epoch_id % 10 == 0:
            acc = predicts.argmax(1) == y.flatten().astype(int)
            acc = acc.astype(float).mean()

            model.eval()
            val_predict = model(paddle.to_tensor(val_data[:, :, :-1])).argmax(1)
            val_label = val_data[:, :, -1]
            val_acc = np.mean(val_predict.numpy() == val_label.flatten())

            print("epoch: {}, iter: {}, loss is: {}, acc is {} / {}".format(
                epoch_id, iter_id, avg_loss.numpy(), acc.numpy(), val_acc))
            
    
model.eval()
test_data = paddle.to_tensor(test_df.values.reshape(-1, 1, 561).astype(np.float32))
test_predict = model(test_data)
test_predict = test_predict.argmax(1).numpy()

test_predict = pd.DataFrame({'Activity': test_predict})
test_predict['Activity'] = test_predict['Activity'].map({
    0:'LAYING',
    1:'STANDING',
    2:'SITTING',
    3:'WALKING',
    4:'WALKING_UPSTAIRS',
    5:'WALKING_DOWNSTAIRS'
})

test_predict.to_csv('submission.csv', index=None)

!zip submission.zip submission.csv
