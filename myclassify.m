function x= myclassify(dataX,filledinx)

option=menu('Choose a classifier:' ,'Associative memory + classifier - hardlim','Associative memory + classifier - purelin','Associative memory + classifier - logsig','Classifier - hardlim','Classifier - purelin','Classifier - logsig') ;

if option == 1
    
    load('AMhardlim.mat');
    
elseif option ==2
    
    load('AMpurelin.mat');
    
elseif option == 3
    
    load('AMlogsig.mat');

elseif option ==4 
    
    load('CLhardlim.mat');

elseif option ==5
    
    load('CLpurelin.mat');
    
elseif option == 6
    
    load('CLlogsig.mat');
   
end

obj=net(dataX);
[a,b]=max(obj);
x=int64(b(filledinx));
    
end