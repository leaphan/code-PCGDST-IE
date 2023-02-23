function [data_s,label_s]  = SHS(data,label,c_min,c_maj,k_noise,k_over,k_smote);
% -----------------------------------------------------
% Function : stage-wise hybrid sampling
% -----------------------------------------------------
% Input
%     data       £ºsubset samples 
%     label      £ºsubset labels
%     c_min      £ºthe minority class
%     c_maj      £ºthe majority class
%     k1         : Number of neighbors for nosiy samples' recognition
%     k2         : Number of neighbors for overlapping recognition
% Output:
%     i_over     : index of overlapping samples
%     i_not_over : index of non-overlapping samples
%     flag       : 1- all majority class samples overlap,0-otherwise
% -----------------------------------------------------

	%% Undersampling based on LORDS
	[~,~,i_over_maj,~,over_flag]  = LORDS(data,label,c_min,c_maj,k_noise,k_over);
	if over_flag == 0 % Eliminate overlapping majority class samples
		data(i_over_maj ,:) = [];
		label(i_over_maj,:) = [];
	end
	
	%% IFO oversampling
	[data_s,label_s] = IFO(data,label,k_smote,over_flag);
end


function [i_over,i_not_over,i_over_maj,i_over_min,flag] = LORDS(data,label,c_min,c_maj,k1,k2)
% -----------------------------------------------------
% Function£ºa local overlapping region dynamic search method based on neighbors
% -----------------------------------------------------
% Input
%     data       £ºsubset samples
%     label      £ºsubset labels
%     c_min      £ºthe minority class
%     c_maj      £ºthe majority class
%     k1         : Number of neighbors for nosiy samples' recognition
%     k2         : Number of neighbors for overlapping recognition
% Output:
%     i_over     : index of overlapping samples
%     i_not_over : index of non-overlapping samples
%     flag       : 1- all majority class samples overlap,0-otherwise
% -----------------------------------------------------
    ntrain = size(data,1);
    flag = 0;

	%% initial the overlapping areas and non-overlapping areas
    % local boundary samples
    label_over = zeros(ntrain,1);
    i_lb = lbInstance(data,label);

    % Nosiy samples
    [idx,~] = knnsearch(data,data,'k',k1+1);
    label_nei = label(idx);
    i_noise = find(mode(label_nei(:,2:end),2) ~= label_nei(:,1));

    % merge
    i_over = unique([meta3judge([],i_lb,[],i_lb);meta3judge([],i_noise,[],i_noise)]);
	label_over(i_over) = 1; 
	i_not_over = setdiff(1:ntrain,i_over); % index of  samples
	
    %% Dynamic search process
    while 1
          [idx,~] = knnsearch(data,data(i_not_over,:),'k',k2+1);
          label_nei = label_over(idx);
          ind_potential = find(mode(label_nei(:,2:end),2)==1); % Potentially overlapping samples (P) in this round

          i_not_over_pre = i_not_over;
          for ind = ind_potential'
                i_cur = i_not_over_pre(ind); % Current non-overlapping samples
                i_nei_over = idx(ind,label_nei(ind,:)==1); % Overlapping samples in nearest neighbors
                center_over = mean(data(i_nei_over,:),1); % Center of Nearest Overlapped Samples
                [~,dis1] = knnsearch(center_over,data(i_cur,:)); 
                [~,dis_all] = knnsearch(center_over,data(i_nei_over,:));
                dis2 = mean(dis_all);
                if dis1 <= dis2 || length(i_nei_over) == k2
					% Update overlapping area
					i_add = i_not_over_pre(ind); % new overlapping samples
					i_not_over = setdiff(i_not_over,i_add); 
					label_over(i_add) = 1; 
                end
          end

          if length(i_not_over_pre) == length(i_not_over) % End the iteration if there are no new overlapping samples
                break;
          end
    end
	
	% The final overlap recognition result
    i_over = setdiff(1:ntrain,i_not_over);
    i_over_maj = intersect(find(label == c_maj),i_over);
    i_over_min = intersect(find(label == c_min),i_over);

    %% Update full overlap sign
    if length(i_over_maj) == length(find(label == c_maj))
          flag = 1; 
    end
end


function [data_new,label_new,data_out] = IFO(data,label,k,over_flag)
% -----------------------------------------------------
% Function£ºIterative filter oversampling
% -----------------------------------------------------
% Input
%     data      £ºsamples
%     label     £ºlabels
%     k         £ºThe number of neighbors
%     over_flag £º1- Majority class samples all overlap,0-otherwise
% Output:
%     data_new  : samples after oversampling
%     label_new : labels after oversampling
%     data_out  : Synthesized samples
% -----------------------------------------------------

    [c_min,c_maj,~,n_min,n_maj] = moreORless(label);
    data_maj = data(label == c_maj,:);
    data_min = data(label == c_min,:);
    

    if n_maj - n_min == 0 % No synthesis required
        data_new = data;
        label_new = label;
        data_out = [];
    elseif n_min == 1 % direct copy
        data_out = repmat(data_min,n_maj-1,1);
        data_new = [data_maj;repmat(data_min,n_maj,1)];
        label_new = [repmat(c_maj,n_maj,1);repmat(c_min,n_maj,1)];
    else
        cnt_syn = 0; % The number of synthesized samples
        data_out = [];
        n_candidate = 1:size(data_min,1);
        while cnt_syn < n_maj - n_min
            %% SMOTE sample synthesis
            n_sel = n_candidate(randperm(length(n_candidate),1)); 
            n_candidate = setdiff(n_candidate,n_sel);
            k_corr = min(size(data_min,1)-1,k);
            idx = knnsearch(data_min,data_min(n_sel,:),'k',k_corr+1);
            n_nei = idx(1+randperm(k_corr,1));
            out_temp = data_min(n_sel,:) + rand(1)*(data_min(n_nei,:) - data_min(n_sel,:)); 
            
            if over_flag == 1 % Conditions not to filter
                data_out = [data_out;out_temp];
                cnt_syn = cnt_syn + 1;
             else
%                 data_out = [data_out;out_temp];
%                 cnt_syn = cnt_syn + 1;
%             end
                %% ENN filtering
                idx_nei = knnsearch(data,out_temp,'k',5);
                if mode(label(idx_nei)) == c_min
                    data_out = [data_out;out_temp];
                    cnt_syn = cnt_syn + 1;
                else
                      data_out = [data_out;out_temp];
                     cnt_syn = cnt_syn + 1;
                end
            end
            
            if length(n_candidate) == 0 
                data_min = [data_min;data_out];
                n_candidate = 1:size(data_min,1);
                data_out = [];
            end
        end
        
        data_new = [data_maj;data_min;data_out];
        label_new = [repmat(c_maj,size(data_maj,1),1);...
            repmat(c_min,size([data_min;data_out],1),1)];
end
end