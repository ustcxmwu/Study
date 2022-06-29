class EventManager(object):

    def __init__(self):
        print("EventManager:: Let me talk to the folks.")

    def arrange(self):
        self.hotalier = Hotelier()
        self.hotalier.book_hotel()

        self.florist = Florist()
        self.florist.set_flower_requirements()

        self.caterer = Caterer()
        self.caterer.set_cuisine()

        self.musician = Musician()
        self.musician.set_music_type()


class Hotelier(object):

    def __init__(self):
        print("Arrangeing the Hotel for Marriage? --")

    def __is_available(self):
        print("Is hotel free for the event on given day?")
        return True

    def book_hotel(self):
        if self.__is_available():
            print("Registered the Booking.\n\n")


class Florist(object):

    def __init__(self):
        print("Flower Decorations for the Event? --")

    def set_flower_requirements(self):
        print("Carnations, Roses and Lilies would be use for decorations\n\n")


class Caterer(object):

    def __init__(self):
        print("Food Arrangements for the Event? --")

    def set_cuisine(self):
        print("Chinese & Coutinenta Cisine to be served.\n\n")


class Musician(object):

    def __init__(self):
        print("Musical Arrangements for the Marriage? --")

    def set_music_type(self):
        print("Jazz and Classical will be played\n\n")


class You(object):
    def __init__(self):
        print("You:: Whoa! Marriage Arrangements??!!!")

    def ask_event_manager(self):
        print("You:: Let's Contact the Event Manager\n\n")
        em = EventManager()
        em.arrange()

    def __del__(self):
        print("You:: Thanks to Event Manager, all preparations done!")


if __name__ == '__main__':
    you = You()
    you.ask_event_manager()
