### Machine-learning Analysis of Elite Selection in the CPC
# Loading packages
library("pROC")
library("caret")
library("caretEnsemble")
library("boot")
library("randomForest")
library("nnet")
library("lmtest")
library("egcm")
library("MASS")
library("tnet")
library("iter")
library("cp")
library("layer1")
library("AUCRF")

# Loading Data - scheduled to be updated every 6 months (e.g., data_cpc_2021_JUN.Rdata)
mydata_train <- get(load("data_train_2020_DEC.Rdata"))
mydata_pred <- get(load("data_POLI_2020_DEC.Rdata"))

# ML list
md_idx1 <- c('glm','glmnet','gbm','rf','svmRadial','nnet','knn') #default model
md_idx2 <- c('glm','treebag','gbm','rf','svmRadial','nnet','knn') #alternative model if glmnet error
md_idx3 <- c('glm','bayesglm','glmnet','ada','pda','treebag','rpart','rf','gbm','svmRadial','nnet','knn','dnn') #full model to decide the right set of ensemble compositions

# ML Function
pred_model <- function(data_train,data_pred,md_idx,posi_idx,pc_idx,AUCRF_idx){
  
  ### Setting
  stime <- Sys.time()
  print(md_idx)
  var_id <- c('promo','ename','pc','year','posi')
  var_list <- names(data_train)[!(names(data_train)%in%var_id)]
  data_train$dv <- data_train$promo
  
  ### Data generation
  dat_tr <- subset(data_train,posi%in%posi_idx&pc%in%pc_idx,select=c('dv',var_id,var_list)) 
  dat_tr <- na.omit(dat_tr)
  dat_pred <- data_pred
  
  print("TRAINING SETS:")
  print(table(dat_tr$year,dat_tr$posi))

  ### Variable pre-selection for efficiency by AUCRF
  if(AUCRF_idx==1){
    print("Executing AUCRF...")
    dat_tr$dv <- factor(dat_tr$dv)
    AUCRF_out <- AUCRF(dv~.,data=dat_tr)
    var_list <- AUCRF_out$Xopt
    print("AUCRF Done!")}
  
  ### ML with full data
  dat0 <- subset(dat_tr,select=c('dv',var_list))
  dat0$dv <- as.factor(ifelse(dat0$dv==1,'YES','NO'))
  dat_control <- trainControl(method="cv", number=5, savePredictions='final', classProbs=TRUE)
  md0_m <- caretList(dv~.,data=dat0,trControl=dat_control,methodList=md_idx)
  en_control <- trainControl(method="boot",number=5,savePredictions="final",classProbs=TRUE,summaryFunction=twoClassSummary)
  en0 <- caretStack(md0_m,method='glm',metric='ROC',trControl=en_control)
  en0_cor <- coef(en0$ens_model$finalModel)[-1]
  en0_pred <- predict(en0,type='prob',se=T)
  pred0 <- lapply(md0_m,predict,type='prob',se=T)
  pred0 <- data.frame(sapply(pred0,function(x) x[,'YES']))
  pred0$en <- en0_pred$fit
  auc_dat0 <- caTools::colAUC(pred0,dat0$dv)
  en0_pred_out <- predict(en0,newdata=data_pred,type='prob',se=T)
  pred0_out <- lapply(md0_m,predict,newdata=data_pred,type='prob',se=T)
  pred0_out <- data.frame(sapply(pred0_out,function(x) x[,'YES']))
  pred0_out$en <- en0_pred_out$fit
  for (var in names(pred0_out)){dat_pred[,paste0('pred0_',var)]<-pred0_out[,var]}
  
  ### ML with cross-validating
  dat <- dat_tr
  dat$group <- sample(1:5,nrow(dat),replace=T)
  dat$dv <- as.factor(ifelse(dat$dv==1,'YES','NO'))
  
  auc_dat1 <- data.frame()
  auc_dat2 <- data.frame()
  
for (i in 1:5){
    print(i)
    dat1 <- subset(dat,group!=i,select=c('dv',var_list))
    dat2 <- subset(dat,group==i,select=c('dv',var_list))
    
    #training with dat1
    dat_control <- trainControl(method="cv", number=5, savePredictions='final', classProbs=TRUE)
    md1_m <- quiet_f(caretList(dv~.,data=dat1,trControl=dat_control,methodList=md_idx))
    en_control <- trainControl(method="boot",number=5,savePredictions="final",classProbs=TRUE,summaryFunction=twoClassSummary)
    en1 <- caretStack(md1_m,method='glm',metric='ROC',trControl=en_control)
    en1_pred <- predict(en1,type='prob',se=T)
    pred1 <- lapply(md1_m,predict,type='prob',se=T)
    pred1 <- data.frame(sapply(pred1,function(x) x[,'YES']))
    pred1$en <- en1_pred$fit
    auc_dat <- caTools::colAUC(pred1,dat1$dv)
    rownames(auc_dat) <- paste0('cv_',i)
    auc_dat1 <- rbind(auc_dat1,auc_dat)

    #testing with dat2 (Test sets)
    en2_pred <- predict(en1,newdata=dat2,type='prob',se=T)
    pred2 <- lapply(md1_m,predict,newdata=dat2,type='prob',se=T)
    pred2 <- data.frame(sapply(pred2,function(x) x[,'YES']))
    pred2$en <- en2_pred$fit
    auc_dat <- caTools::colAUC(pred2,dat2$dv)
    rownames(auc_dat) <- paste0('cv_',i)
    auc_dat2 <- rbind(auc_dat2,auc_dat)
    
    #data_out (prediction for the future data)
    en1_pred_out <- predict(en1,newdata=data_pred,type='prob',se=T)
    pred1_out <- lapply(md1_m,predict,newdata=data_pred,type='prob',se=T)
    pred1_out <- data.frame(sapply(pred1_out,function(x) x[,'YES']))
    pred1_out$en <- en1_pred_out$fit
    for (var in names(pred1_out)){dat_pred[,paste0('pred1_',var,'_cv_',i)]<-pred1_out[,var]}
 }
  
  out <- list(dat_pred=dat_pred,auc0=auc_dat0,auc1=auc_dat1,auc2=auc_dat2,var_list=var_list)
  etime <- Sys.time()
  time <- etime - stime
  print(time)
  return(out)
}

# Iterations
out_list <- list(); for (i in 1:1000){out_list[[i]] <- pred_model(mydata_train,mydata_pred,md_idx1,'Center',c(12:17),0) } #Pre-Xi model
out_list <- list(); for (i in 1:1000){out_list[[i]] <- pred_model(mydata_train,mydata_pred,md_idx2,'Center',c(18),0)} #Xi model
out_list <- list(); for (i in 1:1000){out_list[[i]] <- pred_model(mydata_train,mydata_pred,md_idx1,c('Center','Prov'),c(12:19),1)} #institutional model
