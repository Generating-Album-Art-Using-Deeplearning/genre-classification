import matplotlib.pyplot as plt

f = open('./tmpdata/cnumid.txt', 'r')
lines = f.readlines()
loss_list = []
acc_list = []

for line in lines :
    line = line.strip()
    lst = line.split(' ')
    loss = float(lst[2])
    acc = float(lst[4])

    loss_list.append(loss)
    acc_list.append(acc)

plt.plot(acc_list)
plt.show()

plt.plot(loss_list)
plt.show()