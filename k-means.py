
import math

dataSet = [[1,1],[1,5],[10,85]]

dataClust = 2


dataNum = len(dataSet)





'''  两点间距离   '''
def Distance(node1,node2):
    distance = pow((node1[0] - node2[0]),2) + pow((node1[1] - node2[1]),2)
    return math.sqrt(distance)


def kmeans(dataSet,clust):

    centroids = []
    tempdata = []
    

    #随机添加聚类中心 此处为按顺序取数据集 可改为随机
    for i in range(clust):
        centroids.append(dataSet[i])

    #驯悍聚类
    for i in range(100):
        for i in range(clust):
            tempdata.append([])
        #计算每个样本与聚合中心的距离
        for i in range(len(dataSet)):
            minDistance = Distance(dataSet[i],centroids[0])
            tempType = 0
            for j in range(1,len(centroids)):
                tempDistance = Distance(dataSet[i],centroids[j])
                if tempDistance < minDistance:
                    minDistance = tempDistance
                    tempType = j
            tempdata[tempType].append(dataSet[i])

        print(len(tempdata))
        print(tempdata)

        equalnum = 0
        for i in range(len(centroids)):
            sum = [0,0]
            for j in range(len(tempdata[i])):
                sum[0] = sum[0] + tempdata[i][j][0]
                sum[1] = sum[1] + tempdata[i][j][1]

            sum[0] = sum[0]/len(tempdata[i])
            sum[1] = sum[1]/len(tempdata[i])

            if centroids[i] == sum:
                equalnum +=1

            else:
                centroids[i] = sum

        if equalnum == clust:
            print("equal end!")
            break    
        tempdata.clear()
                  
    print(tempdata)
    print(centroids)  


kmeans(dataSet,dataClust)

