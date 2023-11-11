#加载相关库函数
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
def createDataSet():
    #生成数据集
    dataSet = [[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    #4个特征
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataSet, labels
#特征选择
def calcShannonEnt (dataset):
        #计算给定数据集的熵
        numexamples = len(dataset)  #样本总个数
        labelCounts = {}  # 类别字典，格式｛类别：值｝
        # 统计每个类别的样本个数，存到字典labelCounts中
        for featVec in dataset:
                currentlabel = featVec[-1]
                if currentlabel not in labelCounts.keys():
                        labelCounts[currentlabel] = 0
                labelCounts[currentlabel] += 1
        shannonEnt = 0
        # 计算数据集的熵值
        for key in labelCounts:
                prop = float(labelCounts[key])/numexamples
                shannonEnt -= prop*log(prop,2)
        return shannonEnt

def getDataSet (dataset, featNum, featvalue) :
        #划分数据集，返回第featNum个特征下值为value 的样本集合，
        # #并且返回的样本数据中已经删除给定特征 featNum 和值value
        retDataSet = []
        #创建新的list对象
        for featVec in dataset:
                if featVec[featNum] == featvalue:
                        #从样本中删除第featNum个特征其值value
                        reducedFeatVec = featVec[: featNum]
                        reducedFeatVec.extend(featVec[featNum+1:])
                        retDataSet.append(reducedFeatVec)
        return retDataSet

def chooseBestFeatureToSplit(dataset):
        #选择最好的特征
        featNum = len(dataset[0]) - 1
        # 计算样本熵值，对应公式中：H（D）
        baseEntropy = calcShannonEnt(dataset)
        bestInfoGain = 0
        bestFeature = -1
        # 以每一个特征进行分类，找出使信息增益最大的特征
        for i in range(featNum):
                # 获得该特征的所有取值
                featList = [example[i] for example in dataset]
                # 去掉重复值
                uniqueVals = set(featList)
                newEntropy = 0
                # 计算以第i个特征进行分类后的熵值
                for val in uniqueVals:
                        subDataSet = getDataSet(dataset, i, val)
                        prob = len(subDataSet) / float(len(dataset))
                        # 计算满足第i个特征，值为va1的数据集的熵，并累加该特征熵
                        newEntropy += prob * calcShannonEnt(subDataSet)
                #计算信息增益
                infoGain = baseEntropy - newEntropy
                # 找出最大的熵值及其对应的特征
                if (infoGain > bestInfoGain):
                        bestInfoGain = infoGain
                        bestFeature = i
        return bestFeature

def majoritCnt (classlist):
        classCount={}
        # 统计每个类别的样本个数
        for vote in classlist:
                if vote not in classCount.keys():classCount[vote] = 0
                ClassCount[vote] += 1
        # iteritems：返回列表迭代器
        # operator.itemgeter（1）：获取对象第一个域的值
        # True：降序
        sortedclassCount = sorted(classCount.items(),\
                           key=operator.itemgetter(1), reverse=True)
        return sortedclassCount[0][0]

def createTree(dataset, labels, featLabels):
        #构建决策树
        #classList：数据集的分类类别
        classList = [example[-1] for example in dataset]
        # 所有样本属于同一类时，停止划分，返回该类别
        if classList.count(classList[0]) == len(classList):
                return classList[0]
        # 所有特征已经遍历完，停止划分，返回样本数最多的类别
        if len(dataset[0]) == 1:
                return majorityCnt(classList)
        #选择最好的特征进行划分
        bestFeat = chooseBestFeatureToSplit(dataset)
        bestFeatLabel = labels[bestFeat]
        featLabels.append(bestFeatLabel)
        #以字典形式存储决策树
        myTree = {bestFeatLabel:{}}
        del labels[bestFeat]
        #根据选择特征，遍历所有值，每个划分子集递归调用createDecideTree
        featValue = [example[bestFeat] for example in dataset]
        uniqueVals = set (featValue)
        for value in uniqueVals:
                sublabels = labels[:]
                myTree[bestFeatLabel][value] = createTree\
                        (getDataSet (dataset, bestFeat, value), sublabels, featLabels)
        return myTree

if __name__ == '__main__':
        dataset, labels = createDataSet()
        featLabels = []
        #生成决策树
        myTree = createTree (dataset, labels, featLabels)
        #输出结果
        print (myTree)