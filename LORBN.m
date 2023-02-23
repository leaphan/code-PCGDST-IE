function [trainY_distinguish,n_notoverlap,n_overlap] = OverlapDistinguish_new_distance(trainX,trainY,k)
    ntrain = size(trainX,1);
    
%     mappedX = tsne(trainX);
%     gscatter(mappedX(:,1), mappedX(:,2),trainY,['g' 'b'],'.',30);
%     lgd = legend({'Majority instances', 'Minority instances'});
%     lgd.FontSize = 30;
%     lgd.FontName = 'Times New Roman';
    
    
    %% 边界样本
    trainY_distinguish = zeros(ntrain,1);
    n_link = tomek_link(trainX,trainY);
    
    %% 异常样本
    [idx,~] = knnsearch(trainX,trainX,'k',6);
    label_nei1 = trainY(idx);
    n_ab = find(mode(label_nei1(:,2:end),2) ~= label_nei1(:,1));
    
    %% 初始化重叠区域
    if ~isempty(n_link) && ~isempty(n_ab)
        temp = [n_link(:,1);n_link(:,2);n_ab];
    elseif isempty(n_link) && isempty(n_ab)
        n_notoverlap = 1:ntrain;
        n_overlap = [];
        return;
    elseif ~isempty(n_link)
        temp = [n_link(:,1);n_link(:,2)];
    else
        temp = n_ab;
    end
    n_over_first = unique(temp);
% 
%     tempLabel = trainY;
%     tempLabel(n_over_first,:) = 3;
%     gscatter(mappedX(:,1), mappedX(:,2),tempLabel,['g' 'b' 'r'],'.',30);
%     lgd = legend({'Majority instances', 'Minority instances','Initial overlapping instances'});
%     lgd.FontSize = 30;
%     lgd.FontName = 'Times New Roman';

    n_notoverlap = setdiff(1:ntrain,n_over_first); % 更新非重叠样本索引
    trainY_distinguish(n_over_first) = 1; % 1表示重叠
    
    while 1
        [idx,~] = knnsearch(trainX,trainX(n_notoverlap,:),'k',k+1);% 计算非重叠区域样本的K近邻
        label_nei = trainY_distinguish(idx);
        index_candidate = find(mode(label_nei(:,2:end),2)==1); % 对应的完整索引为n_notoverlap
        %% 找出这些潜在重叠样本的k近邻中的重叠样本（至少三个）
        i_newover = [];
        for ind = 1:length(index_candidate)
            i_self = n_notoverlap(index_candidate(ind)); % 该样本索引
            i_over_nei = idx(index_candidate(ind),label_nei(index_candidate(ind),:)==1); % 近邻重叠索引
            center_nei = mean(trainX(i_over_nei,:),1); % 重叠样本的中心点
            % 该样本到重叠中心的距离
            [~,dis] = knnsearch(center_nei,trainX(i_self,:));
            [idx_temp,dis_temp] = knnsearch(trainX(i_over_nei,:),trainX(i_over_nei,:),'k',length(i_over_nei)+1);
            % 重叠样本的平均间距
            mean_dis_over = sum(sum(dis_temp))/(length(i_over_nei)*(length(i_over_nei)-1));
            if dis < mean_dis_over
                i_newover = [i_newover;index_candidate(ind)];
            end
        end
        if isempty(i_newover) % 无更新则结束寻找
            break;
        else % 有新增重叠区域样本
            index_add = n_notoverlap(i_newover); % 新增的重叠样本索引
            trainY_distinguish(index_add) = 1;
            n_notoverlap = setdiff(n_notoverlap,index_add); % 更新非重叠区域索引
        end
    end
    n_overlap = setdiff(1:ntrain,n_notoverlap);% 最终的重叠样本索引
    
  
%     tempLabel(setdiff(n_overlap,n_over_first'),:) = 4;
%     gscatter(mappedX(:,1), mappedX(:,2),tempLabel,['g' 'b' 'r' 'm'],'.',30);
%     lgd = legend({'Majority instances', 'Minority instances','Initial overlapping instances','New overlapping instances'});
%     lgd.FontSize = 30;
%     lgd.FontName = 'Times New Roman';
    
%     tempLabel = trainY;
%     tempLabel(n_overlap,:) = 3;
%     gscatter(mappedX(:,1), mappedX(:,2),tempLabel,['g' 'b' 'r'],'.',30);
%     lgd = legend({'Majority instances', 'Minority instances','Overlapping instances'});
%     lgd.FontSize = 30;
%     lgd.FontName = 'Times New Roman';
%     a=1;
end