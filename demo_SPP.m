% code is written by Jie Wen
% If any problems, please contact: wenjie@hrbeu.edu.cn
% Please cite the reference:
% Jie Wen, Yong Xu, Zuoyong Li, Zhongli Ma, Yuanrong Xu, 
% Inter-class sparsity based discriminative least square regression [J],
% Neural Networks, 2018, doi: 10.1016/j.neunet.2018.02.002.

clear all
clc
clear memory;

name = 'YaleB_32x32'
load (name);
fea = double(fea);
sele_num = 30;
Eigen_NUM=195;
nnClass = length(unique(gnd));  % The number of classes;
num_Class = [];
for i = 1:nnClass
  num_Class = [num_Class length(find(gnd==i))]; %The number of samples of each class
end
%%------------------select training samples and test samples--------------%%
Train_Ma  = [];
Train_Lab = [];
Test_Ma   = [];
Test_Lab  = [];
for j = 1:nnClass    
    idx      = find(gnd==j);
    randIdx  = randperm(num_Class(j));
    Train_Ma = [Train_Ma; fea(idx(randIdx(1:sele_num)),:)];            % select select_num samples per class for training
    Train_Lab= [Train_Lab;gnd(idx(randIdx(1:sele_num)))];
    Test_Ma  = [Test_Ma;fea(idx(randIdx(sele_num+1:num_Class(j))),:)];  % select remaining samples per class for test
    Test_Lab = [Test_Lab;gnd(idx(randIdx(sele_num+1:num_Class(j))))];
end
Train_Ma = Train_Ma';                       % transform to a sample per column
Train_Ma = Train_Ma./repmat(sqrt(sum(Train_Ma.^2)),[size(Train_Ma,1) 1]);
Test_Ma  = Test_Ma';
Test_Ma  = Test_Ma./repmat(sqrt(sum(Test_Ma.^2)),[size(Test_Ma,1) 1]);  % -------------

label = unique(Train_Lab);
Y = bsxfun(@eq, Train_Lab, label');
Y = double(Y)';
X = Train_Ma;


% the PCA preprocessing step
PCA_Projection = pca(Train_Ma');
PCA_Projection = PCA_Projection(:,1:Eigen_NUM);

%projected onto the pca subspace
Train_SET=PCA_Projection'*Train_Ma; % size of (Eigen_NUM,Train_NUM); % PCA-based 
Test_SET=PCA_Projection'*Test_Ma; 

%SPP method
[W] = SPP(Train_SET,Eigen_NUM);

Train_Maa = W'*Train_SET;
Test_Maa  = W'*Test_SET;
Train_Maa = Train_Maa./repmat(sqrt(sum(Train_Maa.^2)),[size(Train_Maa,1) 1]);
Test_Maa  = Test_Maa./repmat(sqrt(sum(Test_Maa.^2)),[size(Test_Maa,1) 1]);    

Mdl= fitcknn(Train_Maa', Train_Lab,'Distance','euclidean','NumNeighbors',1,'Standardize',1,'BreakTies','nearest');
[class_test] = predict(Mdl,Test_Maa'); 

rate_acc = sum(Test_Lab == class_test)/length(Test_Lab)*100
