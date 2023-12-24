import matplotlib.pyplot as plt

class Meter:
    def __init__(self):
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
    
    def update(self, train_loss, train_acc, val_loss, val_acc):
        self.train_loss_list.append(train_loss)
        self.train_acc_list.append(train_acc)
        self.val_loss_list.append(val_loss)
        self.val_acc_list.append(val_acc)
    
    def plot(self):
        x = range(len(self.train_loss_list))
        plt.figure(figsize=(8, 10))
        
        ax1 = plt.subplot(211)
        ax1.plot(x, self.train_loss_list)
        ax1.plot(x, self.val_loss_list, color='r')
        ax1.legend(['train_loss', 'val_loss'])
        ax1.set_title('loss')
        
        ax2 = plt.subplot(212)
        ax2.plot(x, self.train_acc_list)
        ax2.plot(x, self.val_acc_list, color='r')
        ax2.legend(['train_acc', 'val_acc'])
        ax2.set_title('acc')
        
        plt.savefig('result2.svg')