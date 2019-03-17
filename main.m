%% run this section if you want to replace the neural networks, currently in the folder, by creating new ones


% Matrix creation

model=eye(10);
trainTarget=[]; %Target 10x800

for i=1:80
    trainTarget=horzcat(trainTarget,model);
end

testTarget=[]; %Target test 10x50

for i=1:5
    testTarget=horzcat(testTarget,model);
end


load('PerfectArial');

perfTarget=[];

for i=1:80
    perfTarget=horzcat(perfTarget,Perfect);
end

save('trainTarget','trainTarget') % saved for future use if necessary
save('testTarget','testTarget') % saved for future use if necessary
save('perfTarget','perfTarget') % saved for future use if necessary


% OCR

load ('P.mat') %P 256x800
load('test.mat') %test 256x50

option=menu('What Neural Network do you want to create?' ,'Associative memory + classifier - hardlim','Associative memory + classifier - purelin','Associative memory + classifier - logsig','Classifier - hardlim','Classifier - purelin','Classifier - logsig') ;

if option == 1
    
	W = perfTarget*pinv(P);
    P=W*P;
    Ptest=W*Ptest;
    
    net = perceptron;
    
        %Initialization
        W=rand(10,256);
        b=rand(10,1);
        
        %Training parameters
        net.performParam.lr=0.5;
        net.trainParam.epochs =1000;
        net.trainParam.show =35;
        net.trainParam.goal =1e-6;
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = 1:680;
        net.divideParam.valInd = 680:800;

        
        %Training
        [net,tr]=train(net,P,trainTarget);
        
        %performance (roc curves)
        outputs = net(P);
        errors = gsubtract(outputs,trainTarget);
        performanc = perform(net,trainTarget,outputs);
        plotperf(tr)
        figure
        outputs2=sim(net,Ptest);
        plotroc(trainTarget,outputs)
        figure
        plotroc(testTarget,outputs2)
        sprintf('Train performance')
        performance(outputs,trainTarget)*100
        sprintf('Test performance')
        performance(outputs2,testTarget)*100
        
        AMhardlim=net;
        save AMhardlim


elseif option ==2
    
	W = perfTarget*pinv(P);
    P=W*P;
    Ptest=W*Ptest;

    net = network;
    net.numInputs = 1;
    net.inputs{1}.size = 256;
    net.numLayers = 1;
    net.layers{1}.size = 10;
    net.inputConnect(1) = 1;
    net.biasConnect(1) = 1;
    net.outputConnect(1) = 1;
    net.layers{1}.transferFcn = 'purelin';
    net.inputWeights{1}.learnFcn = 'traingd';
    net.biases{1}.learnFcn = 'traingd';

    net.trainFcn = 'traingd';

        %Initialization
        W=rand(10,256);
        b=rand(10,1);
        
        %Training parameters
        net.performParam.lr=0.5;
        net.trainParam.epochs =1000;
        net.trainParam.show =35;
        net.trainParam.goal =1e-6;
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = 1:680;
        net.divideParam.valInd = 680:800;
        
        %Training
        [net,tr]=train(net,P,trainTarget);
        
        %performance (roc curves)
        outputs = net(P);
        errors = gsubtract(outputs,trainTarget);
        performanc = perform(net,trainTarget,outputs);
        plotperf(tr)
        figure
        outputs2=sim(net,Ptest);
        
        [index1,index2] = max(outputs);
        outputs = zeros(size(outputs,1),size(outputs,2));
        for i=1:length(index2)
          outputs(index2(i),i)=1;
        end
        
        [index1,index2] = max(outputs2);
        outputs2 = zeros(size(outputs2,1),size(outputs2,2));
        for i=1:length(index2)
          outputs2(index2(i),i)=1;
        end
        
        plotroc(trainTarget,outputs)
        figure
        plotroc(testTarget,outputs2)
        sprintf('Train performance(%)')
        performance(outputs,trainTarget)*100
        sprintf('Test performance(%)')
        performance(outputs2,testTarget)*100
        
        AMpurelin=net;
        save AMpurelin
    
elseif option == 3
    
	W = perfTarget*pinv(P);
    P=W*P;
    Ptest=W*Ptest;

    net = network;
    net.numInputs = 1;
    net.inputs{1}.size = 256;
    net.numLayers = 1;
    net.layers{1}.size = 10;
    net.inputConnect(1) = 1;
    net.biasConnect(1) = 1;
    net.outputConnect(1) = 1;
    net.layers{1}.transferFcn = 'logsig';
    net.inputWeights{1}.learnFcn = 'traingd';
    net.biases{1}.learnFcn = 'traingd';

    net.trainFcn = 'traingd';

        %Initialization
        W=rand(10,256);
        b=rand(10,1);

        %Training parameters
        net.performParam.lr=0.5;
        net.trainParam.epochs =1000;
        net.trainParam.show =35;
        net.trainParam.goal =1e-6;
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = 1:680;
        net.divideParam.valInd = 680:800;
        
        %Training
        [net,tr]=train(net,P,trainTarget);
        
        %performance (roc curves)
        outputs = net(P);
        errors = gsubtract(outputs,trainTarget);
        performanc = perform(net,trainTarget,outputs);
        plotperf(tr)
        figure
        outputs2=sim(net,Ptest);
        
        [index1,index2] = max(outputs);
        outputs = zeros(size(outputs,1),size(outputs,2));
        for i=1:length(index2)
          outputs(index2(i),i)=1;
        end
        
        [index1,index2] = max(outputs2);
        outputs2 = zeros(size(outputs2,1),size(outputs2,2));
        for i=1:length(index2)
          outputs2(index2(i),i)=1;
        end
        
        plotroc(trainTarget,outputs)
        figure
        plotroc(testTarget,outputs2)
        sprintf('Train performance(%)')
        performance(outputs,trainTarget)*100
        sprintf('Test performance(%)')
        performance(outputs2,testTarget)*100
        
        AMlogsig=net;
        save AMlogsig
    
elseif option ==4 

    net = perceptron;
    
        %Initialization
        W=rand(10,256);
        b=rand(10,1);
        
        %Training parameters
        net.performParam.lr=0.5;
        net.trainParam.epochs =1000;
        net.trainParam.show =35;
        net.trainParam.goal =1e-6;
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = 1:680;
        net.divideParam.valInd = 680:800;
        
        %Training
        [net,tr]=train(net,P,trainTarget);
        
        %performance (roc curves)
        outputs = net(P);
        errors = gsubtract(outputs,trainTarget);
        performanc = perform(net,trainTarget,outputs);
        plotperf(tr)
        figure
        outputs2=sim(net,Ptest);
        plotroc(trainTarget,outputs)
        figure
        plotroc(testTarget,outputs2)
        sprintf('Train performance')
        performance(outputs,trainTarget)*100
        sprintf('Test performance')
        performance(outputs2,testTarget)*100
        
        CLhardlim=net;
        save CLhardlim
    
elseif option ==5

    net = network;
    net.numInputs = 1;
    net.inputs{1}.size = 256;
    net.numLayers = 1;
    net.layers{1}.size = 10;
    net.inputConnect(1) = 1;
    net.biasConnect(1) = 1;
    net.outputConnect(1) = 1;
    net.layers{1}.transferFcn = 'purelin';
    net.inputWeights{1}.learnFcn = 'traingd';
    net.biases{1}.learnFcn = 'traingd';

    net.trainFcn = 'traingd';

        %Initialization
        W=rand(10,256);
        b=rand(10,1);
        
        %Training parameters
        net.performParam.lr=0.5;
        net.trainParam.epochs =1000;
        net.trainParam.show =35;
        net.trainParam.goal =1e-6;
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = 1:680;
        net.divideParam.valInd = 680:800;
        
        %Training
        [net,tr]=train(net,P,trainTarget);
        
        %performance (roc curves)
        outputs = net(P);
        errors = gsubtract(outputs,trainTarget);
        performanc = perform(net,trainTarget,outputs);
        plotperf(tr)
        figure
        outputs2=sim(net,Ptest);
        
        [index1,index2] = max(outputs);
        outputs = zeros(size(outputs,1),size(outputs,2));
        for i=1:length(index2)
          outputs(index2(i),i)=1;
        end
        
        [index1,index2] = max(outputs2);
        outputs2 = zeros(size(outputs2,1),size(outputs2,2));
        for i=1:length(index2)
          outputs2(index2(i),i)=1;
        end
        
        plotroc(trainTarget,outputs)
        figure
        plotroc(testTarget,outputs2)
        sprintf('Train performance(%)')
        performance(outputs,trainTarget)*100
        sprintf('Test performance(%)')
        performance(outputs2,testTarget)*100
        
        CLpurelin=net;
        save CLpurelin
        
elseif option == 6

    net = network;
    net.numInputs = 1;
    net.inputs{1}.size = 256;
    net.numLayers = 1;
    net.layers{1}.size = 10;
    net.inputConnect(1) = 1;
    net.biasConnect(1) = 1;
    net.outputConnect(1) = 1;
    net.layers{1}.transferFcn = 'logsig';
    net.inputWeights{1}.learnFcn = 'traingd';
    net.biases{1}.learnFcn = 'traingd';

    net.trainFcn = 'traingd';

        %Initialization
        W=rand(10,256);
        b=rand(10,1);
        
        %Training parameters
        net.performParam.lr=0.5;
        net.trainParam.epochs =1000;
        net.trainParam.show =35;
        net.trainParam.goal =1e-6;
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = 1:680;
        net.divideParam.valInd = 680:800;
        
        %Training
        [net,tr]=train(net,P,trainTarget);
        
        %performance (roc curves)
        outputs = net(P);
        errors = gsubtract(outputs,trainTarget);
        performanc = perform(net,trainTarget,outputs);
        plotperf(tr)
        figure
        outputs2=sim(net,Ptest);
        
        [index1,index2] = max(outputs);
        outputs = zeros(size(outputs,1),size(outputs,2));
        for i=1:length(index2)
          outputs(index2(i),i)=1;
        end
        
        [index1,index2] = max(outputs2);
        outputs2 = zeros(size(outputs2,1),size(outputs2,2));
        for i=1:length(index2)
          outputs2(index2(i),i)=1;
        end
        
        plotroc(trainTarget,outputs)
        figure
        plotroc(testTarget,outputs2)
        sprintf('Train performance(%)')
        performance(outputs,trainTarget)*100
        sprintf('Test performance(%)')
        performance(outputs2,testTarget)*100
        
        CLlogsig=net;
        save CLlogsig
   
end

%% run this section to draw and classify (using the neural networks currently in the folder)


run('mpaper.m')