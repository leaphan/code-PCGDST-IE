function [re_data,re_label] = recombination(data,label)
% -----------------------------------------------------
% Function��Disrupt and reorganize the data
% -----------------------------------------------------
% Input��
%     data    ��data
%     label   ��label
% Output:
%     re_data ��reconstituted data
%     re_label��reconstituted label
% -----------------------------------------------------
    n = length(label);
    random_index = randperm(n);
    re_data = data(random_index,:);
    re_label = label(random_index,:);
end