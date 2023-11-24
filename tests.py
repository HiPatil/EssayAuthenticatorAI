from utils import KLDivergence
import numpy as np

def test(mode='normal'):
    kld = KLDivergence(mode=mode)
    p = np.random.randint(0, 100, size=101)
    q = np.random.randint(0, 100, size=101)

    print("The KL divergence is: {}".format(kld.dist(p, q)))

def test2(mode='symmetric'):
    n = 50
    X_test = np.random.randint(0, 100, size = (n, 101))
    y_test = np.random.choice([0, 1], n)

    kld = KLDivergence(mode=mode)
    fit_results = kld.fit(X_test, y_test)

    print(fit_results)

    return fit_results

def test3(mode='symmetric'):
    n = 50
    array_A = np.random.randint(0, 11, size = (n, 101)) # most values between 0 and 10
    #array_B = np.random.choice([*range(11), *range(90, 101)], size=(n, 101)) # some values in 0 to 10 and some in 90 to 100
    array_B = np.random.choice([*range(5, 8), *range(60, 101)], size=(n, 101)) # some values in 0 to 10 and some in 90 to 100
    combined_arr = np.vstack([array_A, array_B])
    y = [1 for _ in range(n)] + [0 for _ in range(n)]
    y = np.array(y)
    kld = KLDivergence(mode=mode)
    fit_results = kld.fit(combined_arr, y)
    print(fit_results)

    print("Testing the predict method")
    y_pred = kld.predict(combined_arr)
    print(y_pred)
    accuracy = np.sum(y_pred == y) / len(y_pred)
    print(f"Accuracy is: {accuracy}")
    return fit_results, accuracy, kld



test('both')

fr = test2()

fr, acc, model = test3()
