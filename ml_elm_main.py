class ML_ELM_:
    def __init__(self, l1, l2, l3, l4):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        pass

        def hidden_nodes(X, input_weights, biases):
            from numpy import dot
            def relu(x):
                from numpy import maximum
                return maximum(x, 0, x)

            G = dot(X, input_weights)
            G = G + biases
            H = relu(G)
            return H

        self.hidden_nodes = hidden_nodes

        def ELM_hidden(train_in, input_weights):
            biases = 0
            hidden = hidden_nodes(train_in, input_weights, biases)

            return hidden

        self.ELM_hidden = ELM_hidden

    def fit(self, train_in_, train_out):
        def ELM_weights(train_in, hidden_size):
            from scipy.linalg import pinv
            from numpy import random
            from numpy import dot
            from numpy import transpose

            input_size = train_in.shape[1]
            hidden_size = int(hidden_size)
            input_weights = random.normal(size=[input_size, hidden_size])
            biases = random.normal(size=[hidden_size])

            output_weights = dot(pinv(self.hidden_nodes(train_in, input_weights, biases)), train_in)
            transposed_weights = transpose(output_weights)

            return transposed_weights

        def ELM_train(train_in, train_out):
            from scipy.linalg import pinv
            from numpy import dot

            output_weights = dot(pinv(train_in), train_out)

            return output_weights

        self.layers = [i for i in list([self.l1, self.l2, self.l3, self.l4]) if i != 0]

        input_layers_tr = [train_in_]

        self.weights = []

        for i, layer in enumerate(self.layers):
            self.weights += [ELM_weights(input_layers_tr[i], layer)]
            input_layers_tr += [self.ELM_hidden(input_layers_tr[i], self.weights[i])]

        self.weights += [ELM_train(input_layers_tr[-1], train_out)]
        return self

    def predict(self, train_in_):
        def predict(input, output_weights):
            from numpy import dot
            out = dot(input, output_weights)
            return out

        def ELM_predict(input, output_weights):
            prediction = predict(input, output_weights)
            return prediction

        input_layers_tr = [train_in_]
        for i, layer in enumerate(self.layers):
            input_layers_tr += [self.ELM_hidden(input_layers_tr[i], self.weights[i])]
        output_train_pred = ELM_predict(input_layers_tr[-1], self.weights[-1])
        return output_train_pred