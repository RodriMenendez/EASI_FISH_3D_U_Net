import torch
import evaluations
import lightning as L

class UNet3DModule(L.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=0.0, loss='dice'):
        super().__init__()
        self.model = model
        self.loss = loss
        self.loss_module = evaluations.DiceLoss() if loss == 'dice' \
                           else torch.nn.CrossEntropyLoss(weight=torch.tensor([0.0008, 0.9992]))
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, inputs):
        return self.model(inputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        output = self.model(inputs)
        output_norm = torch.sigmoid(output).clone() if self.loss == 'dice' else torch.softmax(output, dim=1)
        labels = labels if self.loss == 'dice' else labels.squeeze(1).long()
        loss = self.loss_module(output_norm, labels)

        preds = evaluations.prediction(output_norm)

        confusion_matrix = evaluations.ConfusionMatrix(preds, labels)
        acc = evaluations.accuracy(confusion_matrix)
        prec = evaluations.precision(confusion_matrix)
        iou = evaluations.IoU(confusion_matrix)

        self.log_dict({"train acc": acc, "train precision": prec, "train IoU": iou, "train loss": loss}, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        output = self.model(inputs)
        output_norm = torch.sigmoid(output).clone() if self.loss == 'dice' else torch.softmax(output, dim=1)
        labels = labels if self.loss == 'dice' else labels.squeeze(1).long()
        loss = self.loss_module(output_norm, labels)
        preds = evaluations.prediction(output_norm)

        confusion_matrix = evaluations.ConfusionMatrix(preds, labels)
        acc = evaluations.accuracy(confusion_matrix)
        prec = evaluations.precision(confusion_matrix)
        iou = evaluations.IoU(confusion_matrix)

        self.log_dict({"val acc": acc, "val precision": prec, "val IoU": iou, "val loss": loss})

    def test_step(self, batch, batch_idx):
        inputs, labels = batch

        output = self.model(inputs)
        output_norm = torch.sigmoid(output).clone() if self.loss == 'dice' else torch.softmax(output, dim=1)
        labels = labels if self.loss == 'dice' else labels.squeeze(1).long()
        loss = self.loss_module(output_norm, labels)
        preds = evaluations.prediction(output_norm)

        confusion_matrix = evaluations.ConfusionMatrix(preds, labels)
        acc = evaluations.accuracy(confusion_matrix)
        prec = evaluations.precision(confusion_matrix)
        iou = evaluations.IoU(confusion_matrix)

        self.log_dict({"test acc": acc, "test precision": prec, "test IoU": iou, "test loss": loss})