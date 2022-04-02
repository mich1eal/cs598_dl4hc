# Preprocess original "Cognitive Therapy" study data to feed the Jupyter Notebook for H1.
# H1 is the hypothesis that thought records can be automatically classified into schemas.
# Original code from Franziska Burger, with tweaks:
#    To set working directory
#    To reconstruct line of code that generates training, validation and test datasets
#    To change curly braces to brackets in final write.table call
#    Explicitly output "Utterance" column name for the utterance files

# % Created by Franziska Burger
# % Last modified on: 17.2.2021
# %
# % This is the R script integrated into the manuscript titled "Natural language processing for cognitive therapy: 
# % extracting schemas from thought records" that has been submitted to PLOS ONE in February of 2021. The following files
# % are needed as input files:
# %  -Data/CoreData.csv
# %  -Data/Demographics.csv
# %  -all files in directory Data/IRR
# %  -Data/MentalHealth.csv
# %  -Data/PredictionsH2.csv
# % The following files are produced as ouput files:
# %  -R_Script_integrated_in_paper.pdf (markdown file)
# %  -Data/DatasetsForH1 (training, validation, and test sets for python script to test H1)
# %  -Data/IRR/c1c4/training_50_instructional
# %  -Data/IRR/c1c4/training_set*.csv
# %  -Data/IRR/c1c4/testset.csv
# %  -Figures/schema_trtype.tiff
# %  -Figures/TRdepth.tiff
# %
# % Please note that the working directory needs to be set correctly once at the beginning of this script to the 
# % "AnalysisArticle" directory. 
# % All further paths in the script are relative to this directory.
# %
# % % % % % % % % % % % % % % % % % % % % % % % % 
# %
# % Please contact f.v.burger@tudelft.nl in case you have questions
# %
# % % % % % % % % % % % % % % % % % % % % % % % %
  

# #remove all user-installed libraries in case we still have old 
# #versions
# # create a list of all installed packages
# ip <- as.data.frame(installed.packages())
# head(ip)
# # if you use MRO, make sure that no packages in this library 
# # will be removed
# ip <- subset(ip, !grepl("MRO", ip$LibPath))
# # we don't want to remove base or recommended packages either
# ip <- ip[!(ip[,"Priority"] %in% c("base", "recommended")),]
# # determine the library where the packages are installed
# path.lib <- unique(ip$LibPath)
# # create a vector with all the names of the packages you want 
# # to remove
# pkgs.to.remove <- ip[,1]
# head(pkgs.to.remove)
# # remove the packages
# sapply(pkgs.to.remove, remove.packages, lib = path.lib)

# #install all required libraries
install.packages(c("devtools","githubinstall","plyr",
"dplyr","reshape2","irr","tidyr","ggplot2","magrittr","qdap",
"tm","textclean","tidytext","SnowballC","Hmisc","tidyverse",
"caret","splitstackshape","TeachingDemos","nlme","psych",
"lme4","lmerTest","philentropy","extrafont","lm.beta"))
require(githubinstall)
githubinstall("textclean")

require(plyr)
require(dplyr)
require(reshape2)
#require(irr)
require(tidyr)
#require(ggplot2)
require(magrittr)
require(qdap)
#require(tm)
require(textclean)
require(tidytext)
#require(SnowballC)
require(Hmisc) #for %nin%
require(tidyverse)
require(caret)
require(splitstackshape) #stratified sampling
require(TeachingDemos) #char2seed function
#require(nlme)
#require(psych)
#require(lme4)
#require(lmerTest)
#require(philentropy)
#require(lm.beta)
#require(extrafont)
require(rstudioapi)
#font_import(pattern="[A/a]rial", prompt=FALSE)

# set working directory to data folder (using rstudioapi)
filepath = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(filepath))
#set working directory otherwise
#setwd("~/surfdrive/Documents/Projects/ThoughtRecordChatbot/Experiments/Exploratory_TRs/Documents/DataRepository/AnalysisArticle")

#print session info
sessionInfo()

#set seed for everything to follow
char2seed("Burger",set=TRUE)

#read in the core thought record data
df <- read.csv("Data/CoreData.csv",na.strings = "", 
               header=TRUE, sep=";",fill=TRUE)

schemas <- c("Attach","Comp","Global","Health","Control",
             "MetaCog","Others","Hopeless","OthViews")

#select only relevant columns and rows
df <- df[which(df$UttEnum!="NA"),
         c("Reply",schemas,"Exclude","UttEnum","Scenario",
           "Depth","Participant.ID")] %>% na.omit(.)

#rename Reply column to Utterance
names(df)[names(df) == "Reply"] <- "Utterance"

df[,2:11] <- lapply(df[,2:11], function(x) as.numeric(as.character(x)))

# we remove the exclude sentences from the set
df <- df[which(df$Exclude==0),]
# then we can also remove the exclude column
df$Exclude <- NULL

# we also want a column that says whether the thought 
# record was scenario-based (closed) or a personal one (open)
df$TRtype <- ifelse(df$Scenario=="PTR","open","closed")
df$TRtype <- as.factor(df$TRtype)

#we can examine what impurities are in our text
check_text(df$Utterance)
  
#function that cleans textual utterances. Accepts vector of strings 
#and returns text corpus (indeces are identical)
clean_utts <- function(utts){
  utts <- utts %>% 
    tolower(.) %>% #make everything lower case
    replace_misspelling(.) %>% #try to correct misspelled words
    replace_contraction(.) %>% #expand contractions, can't -> cannot
    replace_number(.) %>% #replace numbers with words
    replace_incomplete(.) %>% #adds/changes missing sentence end
    add_comma_space(.) %>% #adds a space after comma
    rm_stopwords(., strip = TRUE, separate = FALSE)
  #remove excess white spaces
  utts <- sapply(utts, function(x) gsub("\\s+"," ",x)) 
  return(utts)
}

df$Utterance <- clean_utts(df$Utterance)

#sampling function to fulfil three criteria:
# 1. similar distribution of schemas
# 2. approximately the same proportion of open and closed scenarios
# 3. approximately the same distribution over DAT depths as in the entire dataset.

controlled_sampling_H1 <- function(df.pop,perc){
  i<-1 #iteration index
  mse<-1 #the largest possible initial deviation (mean standard error = 1)
  #initialize the sample as the entire population
  selected.sample <- df.pop
  # to long format with schemas labeled as cb
  df.poplong <- df.pop[,c(schemas,"TRtype")] %>% 
    gather(cb,label,1:9)
  # dataframe for distribution over schemas
  df.cbdist <- df.poplong %>% 
    group_by(cb) %>%
    summarise(labeled=sum(label),count=n(), perc=labeled/count)
  # dataframe for distribution over open/closed thought reccords
  df.ocdist <- df.poplong %>% 
    group_by(TRtype) %>%
    summarise(c=n(),perc=c/nrow(df.poplong))
  # dataframe for distribution over DAT depth
  df.depthdist <- df.pop[,c("Depth","TRtype")] %>% 
    group_by(Depth) %>%
    summarise(dcount=n(),perc=dcount/nrow(df.pop))
  while(i<1000){
    #get sample of full population and repeat what we did above
    df.sample <- sample_n(df.pop,ceiling(nrow(df.pop)*(perc/100)))
    df.samplelong <- df.sample[,c(schemas,"TRtype")] %>% 
      gather(cb,label,1:9)
    df.cbdist2 <- df.samplelong %>% group_by(cb) %>%
      summarise(labeled=sum(label),count=n(), perc=labeled/count)
    df.ocdist2 <- df.samplelong %>% group_by(TRtype) %>%
      summarise(c=n(),perc=c/nrow(df.poplong))
    df.depthdist2 <- df.sample[,c("Depth","TRtype")] %>% group_by(Depth) %>%
      summarise(dcount=n(),perc=dcount/nrow(df.pop))
    #now we can compare with population distribution
    comp_cbdist <- mean(abs(df.cbdist$perc-df.cbdist2$perc))
    comp_ocdist <- mean(abs(df.ocdist$perc-df.ocdist2$perc))
    comp_depthdist <- mean(abs(df.depthdist$perc-
                                 df.depthdist2$perc))
    new_mse <- mean(c(comp_cbdist,comp_ocdist,comp_depthdist))
    #update if the new mse is smaller than the previous one
    if(new_mse < mse){
      mse <- new_mse
      selected.sample <- df.sample
    }
    i <- i+1
  }
  return(selected.sample)
}

H1_set_split <- function(df1,df2,perc){ 
  #df1 has binary labels, df2 has ordinal labels
  df.intermediate_test <- controlled_sampling_H1(df1,15)
  df.test <- df2[which(df2$UttEnum %in% df.intermediate_test$UttEnum),]
  df.intermediate_train <- df1[which(df1$UttEnum %nin% 
                                       df.intermediate_test$UttEnum),]
  df.intermediate_validate <- controlled_sampling_H1(
    df.intermediate_train,15)
  df.validate <- df2[which(df2$UttEnum %in% 
                             df.intermediate_validate$UttEnum),]
  df.train <- df[which(df$UttEnum %nin% 
                         df.validate$UttEnum & 
                         df$UttEnum %nin% df.test$UttEnum),]
  sets <- list("train"=df.train,"val"=df.validate,"test"=df.test)
  return(sets)
}

#we recode from the ordinal scale to a binominal one
#to split the set into training, validation, and test set
dfbin <- df
dfbin[,2:10] <- ifelse(dfbin[,2:10] == 2 | dfbin[,2:10]== 3, 1, 0)

# Author left out line of code which creates sets.H1.
# Reconstructing it here.
sets.H1 <- H1_set_split(dfbin, df, 100)

# #we write the sets to a file to save it
write.table(sets.H1$test[,1],"Data/DatasetsForH1/H1_test_texts.csv",
           sep=";",
           col.names=c("Utterance"),  # Added to prevent showing "x" as column name
           row.names=FALSE)
write.table(sets.H1$test[,2:10],"Data/DatasetsForH1/H1_test_labels.csv",
           sep=";",
           row.names=FALSE)
write.table(sets.H1$val[,1],"Data/DatasetsForH1/H1_validate_texts.csv",
           sep=";",
           col.names=c("Utterance"),  # Added to prevent showing "x" as column name
           row.names=FALSE)
write.table(sets.H1$val[,2:10],"Data/DatasetsForH1/H1_validate_labels.csv",
           sep=";",
           row.names=FALSE)
write.table(sets.H1$train[,1],"Data/DatasetsForH1/H1_train_texts.csv",
           sep=";",
           col.names=c("Utterance"),  # Added to prevent showing "x" as column name
           row.names=FALSE)
write.table(sets.H1$train[,2:10],"Data/DatasetsForH1/H1_train_labels.csv",
           sep=";",
           row.names=FALSE)
