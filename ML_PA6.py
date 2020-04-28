'''
    
    PA-6: SVM
    Authors:
    Amitabh Rajkumar Saini, amitabhr@usc.edu
    Shilpa Jain, shilpaj@usc.edu
    Sushumna Khandelwal, sushumna@usc.edu
    
    Dependencies:
    1. numpy : pip install numpy
    2. matplotlib : pip install matplotlib
    3. CVXOPT : pip install cvxopt
    Output:
    Returns a SVM model, writes model parameters on console and generates the plot of the same
    
    '''



import numpy as np
import cvxopt
import cvxopt.solvers


def linear_kernel(x1, x2):
    '''

    :param x1:input matrix x1
    :param x2:input matrix x2
    :return:o/p after applying kernel function of order 1
    '''
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=2):
    '''

    :param x: input matrix x
    :param y: input matrix y
    :param p: denotes the polynomial count set default to 2(quadratic)
    :return: o/p after applying kernel function of order p
    '''
    return (1 + np.dot(x, y)) ** p


class SVM:

    def __init__(self, kernel=linear_kernel, C=None):
        '''

        :param kernel: takes function nam as i/p. It can be linear_kernel or polynomial_kernel
        '''
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        '''

        :param X: input matrix
        :param y: output class matrix
        :return: None
        #calculates bias and weights by using QPP
        '''
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        '''

        :param X: takes input matrix
        :return: hyperplane
        '''
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        '''

        :param X: takes input matrix X
        :return: applies numpy sign function on the o/p of project function y_predict + self.b
        '''
        return np.sign(self.project(X))

    def get_nonlinear_equation(self):
        '''

        :return: returns o/p equation after applying SVM
        '''
        eq = np.zeros(6)
        for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
            eq += a * sv_y * np.asarray([sv[0]**2, sv[1]**2, 1, sv[0]*2, sv[1]*2, 2*sv[0]*sv[1]])
        return eq

    def __str__(self):
        '''
        #Prints necessary outputs on screen
        :return: None
        '''
        print("-------Classifier--------")
        print("Alpha:")
        print(self.a)
        print("Bias:")
        print(self.b)
        if self.kernel == linear_kernel:
            print("Weight:")
            print(self.w)
            print("Center Margin Equation:")
            print(str(self.w[0])+" x1 + "+str(self.w[1])+" x2 + "+str(self.b) + " = 0")
        else:
            print("Weight:")
            print("None")
            print("Center Margin Equation:")
            w = self.get_nonlinear_equation()
            print(str(w[0])+" x1^2+ " + str(w[1]) + " x2^2 + " + str(w[3]) + " x1 + " + str(w[4]) + " x2 + " + str(w[5]) + " x1x2 + " + str(w[2]) + " + " + str(self.b) + " = 0")
        print("Support Vector:")
        print(self.sv)
        print("-------------------------")
        return ""


if __name__ == "__main__":
    import pylab as pl

    def plot_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")
        pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

        # w.x + b = 0
        a0 = 0
        a1 = f(a0, clf.w, clf.b)
        b0 = 1
        b1 = f(b0, clf.w, clf.b)
        pl.plot([a0, b0], [a1, b1], "k")

        # w.x + b = 1
        a0 = 0
        a1 = f(a0, clf.w, clf.b, 1)
        b0 = 1
        b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0, b0], [a1, b1], "k--")

        # w.x + b = -1
        a0 = 0
        a1 = f(a0, clf.w, clf.b, -1)
        b0 = 1
        b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0, b0], [a1, b1], "k--")

        pl.axis("tight")
        pl.show()


    def plot_contour(X1_train, X2_train, clf):
        #Function used for plotting contour with Training Data as inputs
        pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")
        pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-30, 30, 50), np.linspace(-30, 30, 50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()


    def metrics(Y_test, Y_pred):
        #Confusion Matrix calculation and prints the same on screen
        fp = tp = fn = tn = 0
        for i in range(len(Y_pred)):
            expected = Y_test[i]
            if expected == 1:
                if Y_pred[i] == expected:
                    tp += 1
                else:
                    fp += 1

            if expected == -1:
                if Y_pred[i] == expected:
                    tn += 1
                else:
                    fn += 1
        print("----Metrics----")
        print("TP:", tp)
        print("FP:", fp)
        print("TN:", tn)
        print("FN:", fn)
        print("Accuracy:", (tp+tn)/len(Y_pred)*100)
        print("---------------")


    # Linear
    print("-----------LINEAR SVM KERNEL--------------")
    f = np.loadtxt("linsep.txt", delimiter=',')

    x = f[:, :2]
    y = f[:, -1]

    idx = int(len(x) * 0.8)
    xtrain, ytrain = x[:idx], y[:idx]
    xtest, ytest = x[idx:], y[idx:]

    #POSITIVE
    temp = ytrain == 1
    idx = np.arange(len(xtrain))[temp]
    # print(idx)
    # print(idx.shape)
    xtrain_label_pos = xtrain[idx]
    ytrain_label_pos = ytrain[idx]

    # NEGATIVE
    temp = ytrain == -1
    idx = np.arange(len(xtrain))[temp]
    # print(idx)
    # print(idx.shape)
    xtrain_label_neg = xtrain[idx]
    ytrain_label_neg = ytrain[idx]

    clf = SVM()
    clf.fit(xtrain, ytrain)
    pred = (clf.predict(xtest))
    plot_margin(xtrain_label_pos,xtrain_label_neg, clf)
    print(clf)
    metrics(ytest, pred)
    # print(pred)
    # print(ytest)


    #Non Linear
    print("-----------NON LINEAR POLYNOMIAL SVM KERNEL --------------")
    f = np.loadtxt("nonlinsep.txt", delimiter=',')
    np.random.shuffle(f)
    x = f[:, :2]
    y = f[:, -1]
    idx = int(len(x) * 0.8)
    xtrain, ytrain = x[:idx], y[:idx]
    xtest, ytest = x[idx:], y[idx:]

    # POSITIVE
    temp = ytrain == 1
    idx = np.arange(len(xtrain))[temp]
    xtrain_label_pos = xtrain[idx]
    ytrain_label_pos = ytrain[idx]

    # NEGATIVE
    temp = ytrain == -1
    idx = np.arange(len(xtrain))[temp]
    xtrain_label_neg = xtrain[idx]
    ytrain_label_neg = ytrain[idx]

    clf = SVM(polynomial_kernel, C=1000)
    clf.fit(xtrain, ytrain)
    plot_contour(xtrain_label_pos, xtrain_label_neg, clf)
    ypred = (clf.predict(xtest))
    print(clf)
    metrics(ytest, ypred)


    # Test
    # # POSITIVE
    # temp = ytest == 1
    # idx = np.arange(len(xtest))[temp]
    # xtest_label_pos = xtest[idx]
    # ytest_label_pos = ytest[idx]
    #
    # # NEGATIVE
    # temp = ytest == -1
    # idx = np.arange(len(xtest))[temp]
    # xtest_label_neg = xtest[idx]
    # ytest_label_neg = ytest[idx]
    # plot_contour(xtest_label_pos, xtest_label_neg, clf)


