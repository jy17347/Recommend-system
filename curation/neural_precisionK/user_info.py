import numpy as np
class person:
    def __init__(self, name, height, weight, sex, age, ability):
        self.name = name
        self.height = height
        self.weight = weight
        self.sex = sex
        self.age = age
        self.bmi = weight/(height*0.01)**2
        self.ability = ability
    
    def check(self):
        print(self.name, self.height, self.weight, self.sex, self.age, self.bmi, self.ability)

    def profile(self):
        result = np.array([self.name, self.height, self.weight, self.sex, self.age, self.bmi, self.ability])
        result = result[np.newaxis,:]
        return result


def input_bodyInfo():
    name = input("이름: ")
    height = int(input("키: "))
    weight = int(input("몸무게: "))
    sex_str = input("성별 (남, 여): ")
    sex = 1 if sex_str == "남" or 1 else 0
    age = int(input("나이: "))
    ability = int(input("능력: "))
    
    return name, height, weight, sex, age, ability
