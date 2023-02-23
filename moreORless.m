function [c_min,c_maj,IR,n_min,n_maj] = moreORless(label)
    class = unique(label);
    n1 = length(find(label == class(1)));
    n2 = length(find(label == class(2)));
    if n1 >= n2
        c_min = class(2);
        c_maj = class(1);
        IR = n1/n2;
        n_maj = n1;
        n_min = n2;
    else
        c_min = class(1);
        c_maj = class(2);
        IR = n2/n1;
        n_maj = n2;
        n_min = n1;
    end
end