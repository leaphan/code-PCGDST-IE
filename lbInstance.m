function iTarget = lbInstance(data,label)
    [idx,~] = knnsearch(data,data,'K',2,'Distance','euclidean');
    ind = 0;
    flag = 0;
    for i = 1:size(data,1)
        if label(idx(i,1)) ~= label(idx(i,2))
            ind = ind + 1;
            iTarget(ind) = idx(i,1);
            iTarget(ind) = idx(i,2);
            flag = 1;
        end
    end
    if flag == 0
        iTarget = [];
    end
end