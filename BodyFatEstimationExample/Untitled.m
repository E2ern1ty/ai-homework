a=xlsread('in.xlsx','Sheet1');
b=xlsread('tar.xlsx','Sheet1');
c=xlsread('a.xlsx','Sheet1');
d=xlsread('b.xlsx','Sheet1');
inputs = a;
targets = b;
X=inputs;
T=targets;
size(X)
size(T)
net = fitnet(10);
view(net)
[net,tr] = train(net,X,T);
nntraintool
plotperform(tr)
testX =c;
testT = d;
testY = net(testX);
perf = mse(net,testT,testY);
Y = net(X);
plotregression(T,Y)
e = T - Y;
ploterrhist(e)
