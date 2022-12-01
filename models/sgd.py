class SGD:
    def __init__(self, learning_rate, net, loss):
        self.learning_rate = learning_rate
        self.net = net
        self.loss = loss

    def update_rule(self,dW):
        return  dW * self.learning_rate

    def minimize(self, dz_dl=1):
        dz_dl = self.loss.backward(dz_dl)
        for layer in reversed(self.net.config):
            dz_dl = layer.backward(dz_dl)
            layer.update_weights(self.update_rule())
        return input