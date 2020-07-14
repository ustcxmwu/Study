from abc import ABCMeta, abstractmethod


class Subject:
    def __init__(self):
        self.__observers = []

    def register(self, observer):
        self.__observers.append(observer)

    def notify(self, *args, **kwargs):
        for observer in self.__observers:
            observer.notify(self, args, **kwargs)


class ObserverA:
    def __init__(self, sub):
        sub.register(self)

    def notify(self, sub, *args):
        print(type(self).__name__, ':: Got', args, 'From', sub)


class ObserverB:
    def __init__(self, sub):
        sub.register(self)

    def notify(self, sub, *args):
        print(type(self).__name__, ':: Got', args, 'From', sub)


def test_simple_observer():
    subject = Subject()
    observer1 = ObserverA(subject)
    observer2 = ObserverB(subject)
    subject.notify('notification')


class NewsPublisher(object):

    def __init__(self):
        self.__subscribers = []
        self.__lastest_news = None

    def attach(self, subscriber):
        self.__subscribers.append(subscriber)

    def detach(self):
        return self.__subscribers.pop()

    def subscribers(self):
        return [type(subscriber).__name__ for subscriber in self.__subscribers]

    def notify(self):
        for sub in self.__subscribers:
            sub.update()

    def add_news(self, news):
        self.__lastest_news = news

    def get_news(self):
        return "Got news:", self.__lastest_news


class Subscriber(metaclass=ABCMeta):

    @abstractmethod
    def update(self):
        pass


class SMSSubscriber(Subscriber):

    def __init__(self, publisher):
        self.publisher = publisher
        self.publisher.attach(self)

    def update(self):
        print(type(self).__name__, self.publisher.get_news())


class EmailSubscriber(Subscriber):

    def __init__(self, publisher):
        self.publisher = publisher
        self.publisher.attach(self)

    def update(self):
        print(type(self).__name__, self.publisher.get_news())


class AnyOtherSubscriber(Subscriber):

    def __init__(self, publisher):
        self.publisher = publisher
        self.publisher.attach(self)

    def update(self):
        print(type(self).__name__, self.publisher.get_news())


if __name__ == '__main__':
    news_publisher = NewsPublisher()
    for subscriber in [SMSSubscriber, EmailSubscriber, AnyOtherSubscriber]:
        subscriber(news_publisher)

    print("\nSubscribers:", news_publisher.subscribers())

    news_publisher.add_news("Hello World")
    news_publisher.notify()

    print("\nDetached:", type(news_publisher.detach()).__name__)
    print("\nSubscribers:", news_publisher.subscribers())

    news_publisher.add_news("Second News")
    news_publisher.notify()




