function [re_data,re_label] = recombination(data,label)
% -----------------------------------------------------
% Function£ºDisrupt and reorganize the data
% -----------------------------------------------------
% Input£º
%     data    £ºdata
%     label   £ºlabel
% Output:
%     re_data £ºreconstituted data
%     re_label£ºreconstituted label
% -----------------------------------------------------
    n = length(label);
    random_index = randperm(n);
    re_data = data(random_index,:);
    re_label = label(random_index,:);
end