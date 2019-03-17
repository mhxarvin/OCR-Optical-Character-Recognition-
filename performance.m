function x = performance(test,target) % performance calculation
   
    sum=0;
    
    for j=1:size(test,2)
        
        if test(:,j) == target(:,j)
            sum = sum+1;
        end
        
    end
    
    x = sum/size(test,2);
    
end