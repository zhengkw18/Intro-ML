files = ["predict1.csv", "predict2.csv", "predict3.csv", "predict4.csv"]
ys = [0 for _ in range(10000)]
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
    cnt = 0
    for line in lines:
        y = line.split(",")[-1]
        ys[cnt] += float(y)
        cnt += 1
for i in range(10000):
    ys[i] /= len(files)
with open("predict_new.csv", "w", encoding="utf-8") as f:
    lines = ["Id,Predicted\n"]
    for i in range(1, 10000 + 1):
        y = ys[i - 1]
        lines.append("{},{}\n".format(i, float(y)))
    f.writelines(lines)