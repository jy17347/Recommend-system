class person:
    def __init__(self, name, height, weight, sex, age):
        self.name = name
        self.height = height
        self.weight = weight
        self.sex = sex
        self.age = age
    
    def check(self):
        print(self.name)
        print(self.height)
        print(self.weight)
        print(self.sex)
        print(self.age)

    
def input_bodyInfo():
    name = input("이름: ")
    height = int(input("키: "))
    weight = int(input("몸무게: "))
    sex_str = input("성별 (남, 여): ")
    sex = 1 if sex_str == "남" else 0
    age = int(input("나이: "))
    
    return name, height, weight, sex, age
