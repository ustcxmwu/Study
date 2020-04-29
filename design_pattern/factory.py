from abc import ABCMeta, abstractmethod


class Section(metaclass=ABCMeta):
    @abstractmethod
    def describe(self):
        pass


class PersonalSection(Section):
    def describe(self):
        print("Personal Section.")


class AlbumSection(Section):
    def describe(self):
        print("Album Section.")


class PatentSection(Section):
    def describe(self):
        print("Patent Section.")


class PublicationSection(Section):
    def describe(self):
        print("Publication Section.")


class Profile(metaclass=ABCMeta):
    def __init__(self):
        self.sections = []
        self.create_profile()

    @abstractmethod
    def create_profile(self):
        pass

    def get_sections(self):
        return self.sections

    def add_sections(self, section):
        self.sections.append(section)


class linkedin(Profile):
    def create_profile(self):
        self.add_sections(PersonalSection())
        self.add_sections(PatentSection())
        self.add_sections(PublicationSection())


class facebook(Profile):
    def create_profile(self):
        self.add_sections(PersonalSection())
        self.add_sections(AlbumSection())


if __name__=='__main__':
    profile_type = input('Facebook or Linkedin?')
    profile = eval(profile_type.lower())()
    print('Creating Profile..', type(profile).__name__)
    print('profile has sections -- ', profile.get_sections())
