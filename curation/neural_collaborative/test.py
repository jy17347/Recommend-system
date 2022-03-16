from user import get_bodyInfo
from user import person

name, height, weight, sex, age = get_bodyInfo()
newbie = person(name, height, weight, sex, age)
newbie.check()
