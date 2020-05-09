import random
import numpy as np
import sklearn.linear_model as lm
from scipy.stats import f, t
from functools import partial
from pyDOE2 import ccdesign

x_range = [[-4, 4], [-5, 4], [-5, 4]]
x_aver = [sum([x[0] for x in x_range]) / 3, sum([x[1] for x in x_range]) / 3]
y_max = 200 + int(x_aver[1])
y_min = 200 + int(x_aver[0])

def s_kv(y, y_aver, n, m):
    res = []
    for i in range(n):
        s = sum([(y_aver[i] - y[i][j]) ** 2 for j in range(m)]) / m
        res.append(round(s, 3))
    return res

def regression(x, b):
    y = sum([x[i] * b[i] for i in range(len(x))])
    return y

def plan_matrix(n, m):
    print(f'\nПлан матриці при n = {n}, m = {m}')
    y = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m): y[i][j] = random.randint(y_min, y_max)
    no = n - 14 if n > 14 else 1
    x_norm = ccdesign(3, center=(0, no))
    x_norm = np.insert(x_norm, 0, 1, axis=1)
    for i in range(4, 11): x_norm = np.insert(x_norm, i, 0, axis=1)
    l = 1.215
    for i in range(len(x_norm)):
        for j in range(len(x_norm[i])):
            if x_norm[i][j] < -1 or x_norm[i][j] > 1: x_norm[i][j] = -l if x_norm[i][j] < 0 else l

    def add_sq_nums(x):
        for i in range(len(x)):
            x[i][4] = x[i][1] * x[i][2]
            x[i][5] = x[i][1] * x[i][3]
            x[i][6] = x[i][2] * x[i][3]
            x[i][7] = x[i][1] * x[i][3] * x[i][2]
            x[i][8] = x[i][1] ** 2
            x[i][9] = x[i][2] ** 2
            x[i][10] = x[i][3] ** 2
        return x

    x_norm = add_sq_nums(x_norm)
    x = np.ones(shape=(len(x_norm), len(x_norm[0])), dtype=np.int64)
    for i in range(8):
        for j in range(1, 4): x[i][j] = x_range[j - 1][0] if x_norm[i][j] == -1 else x_range[j - 1][1]
    for i in range(8, len(x)):
        for j in range(1, 3): x[i][j] = (x_range[j - 1][0] + x_range[j - 1][1]) / 2
    dx = [x_range[i][1] - (x_range[i][0] + x_range[i][1]) / 2 for i in range(3)]
    x[8][1] = l * dx[0] + x[9][1]
    x[9][1] = -l * dx[0] + x[9][1]
    x[10][2] = l * dx[1] + x[9][2]
    x[11][2] = -l * dx[1] + x[9][2]
    x[12][3] = l * dx[2] + x[9][3]
    x[13][3] = -l * dx[2] + x[9][3]
    x = add_sq_nums(x)
    print('\nX:\n', x)
    print('\nX нормалізоване:\n')
    for i in x_norm: print([round(x, 2) for x in i])
    print('\nY:\n', y)
    return x, y, x_norm


def find_coef(X, Y, norm=False):
    skm = lm.LinearRegression(fit_intercept=False)
    skm.fit(X, Y)
    B = skm.coef_
    print('\nКоефіцієнти рівняння регресії з нормалізованими значеннями X:') if norm == 1 else print('\nКоефіцієнти рівняння регресії:')
    B = [round(i, 3) for i in B]
    print(B)
    print('\nРезультат рівняння зі знайденими коефіцієнтами:\n', np.dot(X, B))
    return B


def cochrane_criterion(y, y_aver, n, m):
    f1 = m - 1
    f2 = n
    q = 0.05
    S_kv = s_kv(y, y_aver, n, m)
    Gp = max(S_kv) / sum(S_kv)
    print('\nПідтвердження критерію Кохрена:')
    return Gp


def cochrane(f1, f2, q=0.05):
    q1 = q / f1
    fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
    return fisher_value / (fisher_value + f1 - 1)

def bs(x, y_aver, n):
    res = [sum(1 * y for y in y_aver) / n]
    for i in range(len(x[0])):
        b = sum(j[0] * j[1] for j in zip(x[:, i], y_aver)) / n
        res.append(b)
    return res


def student_criterion(x, y, y_aver, n, m):
    S_kv = s_kv(y, y_aver, n, m)
    s_kv_aver = sum(S_kv) / n
    s_Bs = (s_kv_aver / n / m) ** 0.5
    Bs = bs(x, y_aver, n)
    ts = [round(abs(B) / s_Bs, 3) for B in Bs]
    return ts


def fischer_criterion(y, y_aver, y_new, n, m, d):
    S_ad = m / (n - d) * sum([(y_new[i] - y_aver[i]) ** 2 for i in range(len(y))])
    S_kv = s_kv(y, y_aver, n, m)
    S_kv_aver = sum(S_kv) / n
    return S_ad / S_kv_aver


def check(X, Y, B, n, m):
    print('\n\tПеревірка рівнянь:')
    f1 = m - 1
    f2 = n
    f3 = f1 * f2
    q = 0.05
    student = partial(t.ppf, q=1 - q)
    t_student = student(df=f3)
    G_kr = cochrane(f1, f2)
    y_aver = [round(sum(i) / len(i), 3) for i in Y]
    print('\nСереднє значення Y:', y_aver)
    disp = s_kv(Y, y_aver, n, m)
    print('Y дисперсія:', disp)
    Gp = cochrane_criterion(Y, y_aver, n, m)
    print(f'Gp = {Gp}')
    if Gp < G_kr:
        print(f'Дисперсії однорідні з вірогідністю: {1 - q}.')
    else:
        print("Збільшити кількість експериментів.")
        m += 1
        main(n, m)
    ts = student_criterion(X[:, 1:], Y, y_aver, n, m)
    print('\nКритерій Стьюдента:\n', ts)
    res = [t for t in ts if t > t_student]
    final_k = [B[i] for i in range(len(ts)) if ts[i] in res]
    print('\nКоефіцієнти {} статично незначні, тому ми видаляємо їх з отриманого рівняння.'.format([round(i, 3) for i in B if i not in final_k]))
    y_new = []
    for j in range(n): y_new.append(regression([X[j][i] for i in range(len(ts)) if ts[i] in res], final_k))
    print(f'\nЗначення Y з коефіцієнтами: {final_k}')
    print(y_new)
    d = len(res)
    if d >= n:
        print('\nF4 <= 0')
        print('')
        return
    f4 = n - d
    F_p = fischer_criterion(Y, y_aver, y_new, n, m, d)
    fisher = partial(f.ppf, q=0.95)
    f_t = fisher(dfn=f4, dfd=f3)
    print('\nПеревірка адекватності за критерієм Фішера')
    print('Fp =', F_p)
    print('F_t =', f_t)
    if F_p < f_t:
        print('93mMath модель адекватна експериментальним даним.')
        return False
    else:
        print('93mMath модель не є адекватною експериментальним даним. Використання квадратичних коефіцієнтів...')
        return True

if __name__ == '__main__':
    n = 25
    m = 3
    test_q = 0
    quadr_q = 0
    for i in range(20):
        X5, Y5, X5_norm = plan_matrix(n, m)
        y5_aver = [round(sum(i) / len(i), 3) for i in Y5]
        B5 = find_coef(X5, y5_aver)
        quadr_q += check(X5_norm, Y5, B5, n, m)
        test_q += 1
    print(f'Квадратичні значення були використані в {round(quadr_q / test_q * 100, 2)}% випадків.')
