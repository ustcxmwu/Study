
quotes = ("A man is not complete until the is married. Then he is finished.",
          "As I said before, I never repeat myself.",
          "Behind a successful man is an exhausted woman.",
          "Black holes really suck...",
          "Facts are stubborn things.")


class QuteModel(object):

    def get_quote(self, n):
        try:
            value = quotes[n]
        except IndexError as err:
            value = "Not found!"
        return value


class QuotoTerminalView(object):

    def show(self, quote):
        print("And the quote is : '{}'".format(quote))

    def error(self, msg):
        print("Error: {}".format(msg))

    def select_quote(self):
        return input("Which quote number would you like to see?")


class QuoteTerminalController(object):
    def __init__(self):
        self.model = QuteModel()
        self.view = QuotoTerminalView()

    def run(self):
        valid_input = False
        while not valid_input:
            try:
                n = self.view.select_quote()
                n = int(n)
                valid_input = True
            except ValueError as err:
                self.view.error("Incorrect index '{}'".format(n))
        quote = self.model.get_quote(n)
        self.view.show(quote)


def main():
    controller = QuoteTerminalController()
    while True:
        controller.run()


if __name__ == "__main__":
    main()
