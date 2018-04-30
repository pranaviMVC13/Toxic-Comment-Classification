import csv
import numpy as np
import Config
def load_data(filename):
    lines = open(filename, 'r').readlines()
    lines = lines[1:] #removing the header

    tokens = []
    data = []
    labels = []
    for line in lines:
        tempLabel = line.split("\t")[3] #taking toxicity score
        if((tempLabel=="False") or (tempLabel=="True")):
            continue;
        data.append(line.split("\t")[1])

        if (float(tempLabel)>=-2.0) & (float(tempLabel)<-1.0):
            tempLabel = [1,0,0,0];
        elif (float(tempLabel)>=-1.0) & (float(tempLabel)<0.0):
            tempLabel = [0,1,0,0];
        elif (float(tempLabel)>=0.0) & (float(tempLabel)<1.0):
            tempLabel = [0,0,1,0];
        elif (float(tempLabel) >= 1.0) & (float(tempLabel)<=2.0):
            tempLabel = [0,0,0,1];
        labels.append(tempLabel)


    if(filename.find("dev")!=-1):
        data = data[:int(0.005*len(data))]
        labels = labels[:int(0.005*len(labels))]
    # print len(data)
    # print len(labels)
    return [data,labels]
# load_data('data/train.csv')

def batch_iteration(data,batch_size,no_epochs,shuffle=True):

    # explicitly feeding keep_prob as well
    # feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels, self.keep_prob: 0.5}

    # data=np.array(data);
    # dataSize = len(data);
    # data = list(zip(trainFeats, trainLabels))
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/Config.batch_size)+1;
    batchTrainList = []
    # if shuffle:
    #     shuffle_indices = np.random.permutation(np.arange(data_size))
    #     shuffled_data = data[shuffle_indices]
    # else:
    #     shuffled_data = data

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]

    # for epoch in xrange(num_batches_per_epoch):
    #     # for batchNum in xrange(numBatchesPerEpoch):
    #         start = (epoch * batch_size) % len(trainFeats)
    #         end = ((epoch + 1) * batch_size) % len(trainFeats)
    #         print "start is:",start;
    #         print "end is:",end;
    #         if end < start:
    #             start -= end
    #             end = len(trainFeats)
    #         batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]
    #         batchTrainList.append([batch_inputs,batch_labels])
    print "bliiiiiiiiiiiiiiiiiiiiiii"

    # return batchTrainList;
            # yield data[startIndex:endIndex];






