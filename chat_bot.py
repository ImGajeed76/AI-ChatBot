from neural_network import *


class Conversation:
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output


class ChatBot:
    message_list_len: int
    conversations: list[Conversation] = []

    def __init__(self, message_list_len=3, max_message_len=100, save_file="chat_bot_network.nn",
                 conversation_file="chat_bot_convs.cb"):
        self.nn = NeuralNetwork()
        self.message_list_len = message_list_len
        self.max_message_len = max_message_len
        self.last_messages = [""] * self.message_list_len

        self.nn.load_network([
            Dense(self.message_list_len * self.max_message_len, self.message_list_len * self.max_message_len),
            TanH(),
            Dense(self.message_list_len * self.max_message_len, self.message_list_len * self.max_message_len * 2),
            TanH(),
            Dense(self.message_list_len * self.max_message_len * 2, self.message_list_len * self.max_message_len),
            TanH(),
            Dense(self.message_list_len * self.max_message_len, self.max_message_len * 3),
            TanH(),
            Dense(self.max_message_len * 3, self.max_message_len * 2),
            TanH(),
            Dense(self.max_message_len * 2, self.max_message_len),
            TanH()
        ])

        self.save_file = save_file
        self.conversation_file = conversation_file

    def load_network_from_file(self, file: str = None):
        if file is not None:
            if os.stat(file).st_size == 0:
                return
            self.nn.load_network_from_file(file)
        else:
            if os.stat(self.save_file).st_size == 0:
                return
            self.nn.load_network_from_file(self.save_file)

    def load_network(self, network):
        self.nn.load_network(network)

    def text_to_floats(self, text: str):
        float_list = []
        for c in text:
            float_list.append(ord(c) / 256)

        for i in range(self.max_message_len - len(text)):
            float_list.append(32 / 256)

        return float_list

    def floats_to_text(self, floats: list[list[float]]):
        out = ""
        for f in floats:
            out += chr(int(f[0] * 256) % 256)

        return out

    def trim_message(self, message: str):
        out = ""

        for i in range(len(message)):
            if i < self.max_message_len:
                out += message[i]

        return out

    def push_message(self, message: str):
        self.last_messages[0] = ""

        for i in range(self.message_list_len - 1):
            self.last_messages[i] = self.last_messages[i + 1]

        self.last_messages[len(self.last_messages) - 1] = self.trim_message(message.lower())

    def get_answer(self):
        inputs = []

        for m in self.last_messages:
            fa = self.text_to_floats(m)
            for f in fa:
                inputs.append([f])

        output = self.nn.forward(inputs)
        return self.floats_to_text(output)

    def train(self, iterations=10000):
        self.load_conversations_from_file()
        print("Training..")
        for i in range(iterations):
            error = 0

            for con in self.conversations:
                eo_floats = []

                fa = self.text_to_floats(con.output)
                for f in fa:
                    eo_floats.append([f])

                inputs = []

                for m in con.inputs:
                    fa = self.text_to_floats(m)
                    for f in fa:
                        inputs.append([f])

                error += self.nn.epoch(np.array(inputs), np.array(eo_floats))

            error /= len(self.conversations)
            print('', end='\r')
            print(
                f'{round((i + 1) / iterations * 100, 2)}% of {iterations} epochs, '
                f'accuracy = {error} = {round((2 - error) / 2 * 100, 2)}%',
                end='')

            self.save_network_to_file(log=False)

        print('\r')
        print('Finished the training')

        self.save_conversations_to_file()
        self.save_network_to_file()

    def epoch(self, expected_output: str, iterations=10):
        self.load_conversations_from_file(log=False)
        self.conversations.append(Conversation(self.last_messages, expected_output))
        error = 10

        for i in range(iterations):
            for con in self.conversations:
                eo_floats = []

                fa = self.text_to_floats(con.output)
                for f in fa:
                    eo_floats.append([f])

                inputs = []

                for m in con.inputs:
                    fa = self.text_to_floats(m)
                    for f in fa:
                        inputs.append([f])

                error = self.nn.epoch(np.array(inputs), np.array(eo_floats))

        print(f"Accuracy = {error} = {round((2 - error) / 2 * 100, 2)}%")
        self.save_conversations_to_file(log=False)
        self.save_network_to_file(log=False)
        return expected_output

    def save_conversations_to_file(self, log=True):
        if log:
            print("Saving..", end='\r')
        with open(self.conversation_file, 'wb') as file:
            pickle.dump(self.conversations, file, protocol=pickle.HIGHEST_PROTOCOL)
        if log:
            print(f"Saved in {self.conversation_file}")

    def load_conversations_from_file(self, log=True):
        if os.stat(self.conversation_file).st_size == 0:
            return

        if log:
            print("Loading..", end='\r')
        with open(self.conversation_file, 'rb') as file:
            conversations = pickle.load(file)
            self.conversations = conversations
        if log:
            print(f"Loaded from {self.conversation_file}")

    def save_network_to_file(self, file: str = None, log=True):
        if file is None:
            self.nn.save_network_to_file(self.save_file, log=log)
        else:
            self.nn.save_network_to_file(file, log=log)
