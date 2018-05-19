setwd('~/unet/U-net/Unet/DataSets/training')
setwd('~/uns2/raw/train/')
require('dplyr')



files<-list.files()
length(files)
fls<-gsub("sample","mask",files) %>%
  table
names(fls[which(fls!=2)])

file.remove(names(fls[which(fls!=2)]))
file.remove(gsub('mask','sample',names(fls[which(fls!=2)])))


samples<-grep('sample',files,value = T)
masks<-grep('mask',files,value = T)

samples %>%
  length
masks %>%
  length

masknewnames<-paste0('im',1:length(masks),'_label.png')
samplenewnames<-paste0('im',1:length(masks),'_color.png')
cbind(masks,samples)

file.rename(from = masks, to = masknewnames)
file.rename(from = samples, to = samplenewnames)
file.de

library('bmp')


label_files<-paste0('./',list.files('./')) %>% grep('mask',.,value=T)
samples_files<-paste0('./',list.files('./')) %>% grep('sample',.,value=T)
masks<-lapply(label_files,read.bmp)
samples<-lapply(samples_files,read.bmp)
str(masks)
hist(samples[[1]])
percent_shelter<-lapply(masks,function(x){sum(x)/length(x)}) %>% unlist
hist(percent_shelter)
goodsamples<-which(percent_shelter>0.01*255)
goodsamples %>%
  length
label_files[badsamples]
badsamples<-which(percent_shelter<0.05)
file.remove(label_files[badsamples])
file.remove(samples_files[badsamples])

percent_shelter[-badsamples] %>%
  mean
setwd('../test/')
testfilenames<-list.files()
file.rename(from = testfilenames,to = substr(testfilenames,3,nchar(testfilenames)))

# masks<-masks %>% lapply(function(x){
#   x[x==0]<-0.5
#   x
# })
# masks<-masks %>%
#   lapply(function(x){x[1,1]<-0
#                      x})
setwd("../training2/")
goodsamples
sample_files_jpg<-paste0(substr(samples_files,1, nchar(samples_files)-4),'.jpg')
labels_files_jpg<-paste0(substr(label_files,1, nchar(label_files)-4),'.jpg')
labels_files_jpg
for(i in goodsamples){
  writeJPEG(image = samples[[i]],target = sample_files_jpg[i])
  writeJPEG(image = masks[[i]],target = labels_files_jpg[i])
}


getwd()
goodsamples
plot(ps[[1]])

lapply(ps,max) %>% unlist %>% max


ps[[1]]
image(p)
min(p)
library("png")
library("jpeg")

# more focused samples

image(labels[[5]])


labels[[1]]

