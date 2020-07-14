
class Model(object):

    def logic(self):
        data = "Got it"
        print("Model: Crunching data as per business logic")
        return data


class View(object):

    def update(self, data):
        print("View: Updating the view with results: {}".format(data))


class Controller(object):

    def __init__(self):
        self.model = Model()
        self.view = View()

    def interface(self):
        print("Controller: Relayed the Client asks")
        data = self.model.logic()
        self.view.update(data)


if __name__ == '__main__':
    print("Client: asks for certain information")
    controller = Controller()
    controller.interface()
    
