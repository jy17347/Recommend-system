from tensorflow.keras import losses
class baysian(losses.Loss):
    def BPRLoss(self, y_true, y_pred,smooth):
        return False



# class BPRLoss(nn.Module):
#     def __init__(self, **kwargs):
#         super(BPRLoss, self).__init__(**kwargs)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, positive, negative):
#         distances = positive - negative
# 		loss = torch.sum(torch.log(self.sigmoid(distances)))
#         return loss
