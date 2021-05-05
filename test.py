class Suri:

    def pretty_sum_ab(func):
        print('asdasd')
        def inner(a, b):
            print(str(a) + " + " + str(b) + " is ", end="")
            return func(a, b)

        return inner


    @pretty_sum_ab
    def sum_ab(a, b):
        summed = a + b
        print(summed)


if __name__ == "__main__":
    s = Suri()