# Partially based on the https://github.com/maximemeylan/Meylan_et_al_2022/

# Seurat 4.3.0 was used for SCTransform v2

# To use MAST for DEA, need another version of Seurat (4.1.1)
# Bug in newer version: invalid name for slot of class “BayesGLMlike”: norm.method
# library(remotes)
# # install dependency which is no longer available on CRAN
# install.packages(c("spatstat.core"), 
#                  repos = "https://spatstat.r-universe.dev")
# remotes::install_version("Seurat", version = "4.1.1") # for MAST

# Bug in 4.1.1 https://github.com/satijalab/seurat/issues/6179
# remotes::install_version("Seurat", version = "4.3.0",lib=PATH_RESULTS) # to plot predictive patches

# Load another version of package
# detach("package:Seurat", unload=TRUE) # if already loaded the old version
# library(Seurat, lib.loc=PATH_RESULTS)

# Additional packages that you may need to install:
# install.packages("hdf5r", dependencies = T)
# if (!require("BiocManager", quietly = TRUE))
#    install.packages("BiocManager")
# BiocManager::install("glmGamPoi")
# BiocManager::install("MAST")

library(Seurat)
library(dplyr)
library(crayon)
library(ggplot2)
library(RColorBrewer)
library(gridExtra)
library(MCPcounter)

res_folder <- paste0(PATH_RESULTS, "/HCC4_ABRS/")

slide_list <- c("D2_count",
                "E4_count",
                "E6_count",
                "E7_count"
)
spatial_list <- sapply(slide_list,function(slide){
  print(slide)
  raw_data_directory <- paste0(PATH_RESULTS, "/",slide,"/outs/")
  spatial_object <- Seurat::Load10X_Spatial(raw_data_directory)
  
  # Mitochondrial genes were excluded from Visium FFPE v1 human probe set
  # Details: https://kb.10xgenomics.com/hc/en-us/articles/4402703463565-How-were-the-genes-included-in-the-Visium-for-FFPE-human-probe-set-chosen-
  genes_to_keep <- names(which(Matrix::rowSums(spatial_object@assays$Spatial@counts )>5))
  
  spatial_object_subset <- subset(spatial_object,features =genes_to_keep, subset = nFeature_Spatial > 300)
  cat("Spots removed: ", ncol(spatial_object) - ncol(spatial_object_subset), "\n")
  cat("Genes kept: ", length(genes_to_keep),"from",nrow(spatial_object), "\n") 
  
  spatial_object_subset <- SCTransform(spatial_object_subset, vst.flavor='v2', assay = "Spatial", verbose = T)
  
  return(spatial_object_subset)
})
save(spatial_list, file=paste0(res_folder,Sys.Date(),"_","seurat_processedv2.RData"))

# load(paste0(res_folder,"2023-04-03_seurat_processedv2.RData"))
spatial_list <- list(D2 = spatial_list[[1]], E4 = spatial_list[[2]], E6 = spatial_list[[3]],
                     E7 = spatial_list[[4]])

gene_signature <- 'ABRS'
geneS = c("CXCR2P1","ICOS","TIMD4","CTLA4","PAX5","FCRL3","AIM2","GBP5","CCL4")

for (i in c(1:length(spatial_list))) {
  print(geneS %in% rownames(spatial_list[[i]][["SCT"]]@data)) # only "CXCR2P1" not found
  print(paste0("Sample number: ", i, ' -> ABRS genes found: ', sum(geneS %in% rownames(spatial_list[[i]][["SCT"]]@data))))
}


# Draw heatmaps for ABRS
for (ids in c(1:length(spatial_list))) {
  genes <- geneS[geneS %in% rownames(spatial_list[[ids]][["SCT"]]@data)] # drop missing gene(s)
  # colors <- c("#053061","#F7F6F6","#67001F")
  
  spatial_list[[ids]] <- AddMetaData(spatial_list[[ids]],apply(as.matrix(spatial_list[[ids]][["SCT"]]@data[genes,]),2,mean),col.name = "ABRS")
  write.csv(as.matrix(spatial_list[[ids]][["SCT"]]@data[genes,]), file=paste0(res_folder, names(spatial_list)[ids], '_ABRS.csv')) #spatial_list[[ids]]$ABRS

  if(is.nan(sum(spatial_list[[ids]]$ABRS))) next
  
  # q80
  jpeg(file=paste0(res_folder,Sys.Date(),"_ABRS_",names(spatial_list)[ids],"_q80.jpeg"),width=960,height=960,quality = 100)
  print(SpatialPlot(spatial_list[[ids]],image.alpha = 0, features = "ABRS", max.cutoff = 'q80') &
          # ggplot2::scale_fill_gradientn(colors=colors, values=scales::rescale(c(-1,0,1))) &
          ggplot2::scale_fill_gradientn(colors=rev(brewer.pal(11,"RdBu"))) & # same as in python
          # ggplot2:: scale_fill_distiller(palette = "Spectral") &
          theme(legend.key.size = unit(3, 'cm'), #change legend key size
                legend.key.height = unit(3, 'cm'), #change legend key height
                legend.key.width = unit(3, 'cm'), #change legend key width
                legend.title = element_text(size=30), #change legend title font size
                legend.text = element_text(size=30))) #change legend text font size
  dev.off()
  
  # q90
  jpeg(file=paste0(res_folder,Sys.Date(),"_ABRS_",names(spatial_list)[ids],"_q90.jpeg"),width=960,height=960,quality = 100)
  print(SpatialPlot(spatial_list[[ids]],image.alpha = 0, features = "ABRS", max.cutoff = 'q90') &
          # ggplot2::scale_fill_gradientn(colors=colors, values=scales::rescale(c(-1,0,1))) &
          ggplot2::scale_fill_gradientn(colors=rev(brewer.pal(11,"RdBu"))) & # same as in python
          # ggplot2:: scale_fill_distiller(palette = "Spectral") &
          theme(legend.key.size = unit(3, 'cm'), #change legend key size
                legend.key.height = unit(3, 'cm'), #change legend key height
                legend.key.width = unit(3, 'cm'), #change legend key width
                legend.title = element_text(size=30), #change legend title font size
                legend.text = element_text(size=30))) #change legend text font size
  dev.off()
  
  # no max cutoff
  jpeg(file=paste0(res_folder,Sys.Date(),"_ABRS_",names(spatial_list)[ids],".jpeg"),width=960,height=960,quality = 100)
  print(SpatialPlot(spatial_list[[ids]],image.alpha = 0, features = "ABRS") &
          # ggplot2::scale_fill_gradientn(colors=colors, values=scales::rescale(c(-1,0,1))) &
          ggplot2::scale_fill_gradientn(colors=rev(brewer.pal(11,"RdBu"))) & # same as in python
          # ggplot2:: scale_fill_distiller(palette = "Spectral") &
          theme(legend.key.size = unit(3, 'cm'), #change legend key size
                legend.key.height = unit(3, 'cm'), #change legend key height
                legend.key.width = unit(3, 'cm'), #change legend key width
                legend.title = element_text(size=30), #change legend title font size
                legend.text = element_text(size=30))) #change legend text font size
  dev.off()
}

# *******************************************************************************
# Differential expression analysis
# Use script select_topN.py
score_type <- "weighted_pred" # weighted_pred 
norm <- 'rescale01'
n_patch <- 100
res_folder_1 <- "PATH_CLAM/eval_results_tcga-349_tumor_masked_multi-output_regression_patch/EVAL_mo-reg_visium-integrate_st_hcc_tumor-masked_ctranspath-tcga-paip_4_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv/ensembled-aver_"
res_folder_2 <- paste0(res_folder, "MAST_",score_type,"_",norm, '_patch_', n_patch)
if (!dir.exists(res_folder_2)) {dir.create(res_folder_2)}
for (i in c(1:length(spatial_list))) {
  dl_class <- read.csv(paste0(res_folder_1,score_type,"_scores_10f_",norm,"/",
                              names(spatial_list)[i], '_rot90_G', n_patch, '.csv'))
  dl_class <- dl_class[,c(-1)]
  spatial_list[[i]] <- AddMetaData(object= spatial_list[[i]],metadata=setNames(dl_class[,paste0('G',n_patch)], dl_class[,'barcode']),col.name = paste0('G',n_patch))
}

# Plot where are this predictive patches
# load seurat version 4.3.0
detach("package:Seurat", unload=TRUE)
library(Seurat, lib.loc=PATH_RESULTS)
for (ids in c(1:length(spatial_list))) {
  jpeg(file=paste0(res_folder_2,"/",Sys.Date(),"_dl_predictive_patches_",names(spatial_list)[ids],".jpeg"),width=960,height=960,quality = 100)
  # this function is not working with Seurat 4.1.1 for group other than ident
  print(SpatialPlot(spatial_list[[ids]], group.by=c(paste0("G",n_patch)),cols=c("Low" = "blue", "Non_Info" = "grey90", "High" = "red")))
  # ggplot2::scale_fill_gradientn(colors=colors, values=scales::rescale(c(-1,0,1))) &
  # ggplot2::scale_fill_gradientn(colors=brewer.pal(11,"Spectral")) &
  # ggplot2:: scale_fill_distiller(palette = "Spectral") &
  # theme(legend.key.size = unit(3, 'cm'), #change legend key size
  #       legend.key.height = unit(3, 'cm'), #change legend key height
  #       legend.key.width = unit(3, 'cm'), #change legend key width
  #       legend.title = element_text(size=30), #change legend title font size
  #       legend.text = element_text(size=30)) #change legend text font size
  dev.off()
}

# back to seurat version 4.1.1
detach("package:Seurat", unload=TRUE)
library(Seurat)

# Plot the distribution
for (ids in c(1:length(spatial_list))) {
  if(is.nan(sum(spatial_list[[ids]]$ABRS))) next
  
  jpeg(file=paste0(res_folder,Sys.Date(),"_ABRS_VlnPlot_G100_",names(spatial_list)[ids],".jpeg"),width=480,height=480,quality = 100)
  print(VlnPlot(subset(spatial_list[[ids]], subset = G100 == 'High' | G100 == 'Low'), features = "ABRS", group.by = 'G100', pt.size=0, cols = c("Low" = "#053061", "High" = "#67001F")) +
          geom_boxplot(width=0.03, alpha = 1, color='black', fill='white', outlier.shape=NA) +
          xlab("") +
          ylab("") +
          labs(title = NULL) +
          theme(legend.position = "none"))
          # theme(axis.text.x = element_blank(), axis.ticks = element_blank()))
  dev.off()
  
  jpeg(file=paste0(res_folder,Sys.Date(),"_ABRS_DotPlot_G100_",names(spatial_list)[ids],".jpeg"),width=960,height=960,quality = 100)
  DATASET <- data.frame(ABRS=subset(spatial_list[[ids]], subset = G100 == 'High' | G100 == 'Low')$ABRS, G100=subset(spatial_list[[ids]], subset = G100 == 'High' | G100 == 'Low')$G100)
  print(ggplot(DATASET, aes(x=G100, y=ABRS, G100)) +
          geom_boxplot(width=0.5, alpha = 1, outlier.shape=NA, fill = c("red", "blue")) +
          xlab("") +
          ylab("") +
          labs(title = NULL) +
          theme(legend.position = "none", panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                panel.background = element_blank(),axis.line.x =element_line(colour = "black"),axis.line.y = element_line(colour = "black"),
                axis.text = element_text(size = rel(1.2)),
                axis.text.x = element_text(angle = 45, hjust=1, colour='black')) +
          geom_dotplot(binaxis = 'y', binwidth=0.001, stackdir = "center")
          # theme(axis.title.x=element_blank(),
          #       axis.text.x=element_blank(),
          #       # axis.ticks.x=element_blank(),
          #       axis.title.y=element_blank(),
          #       axis.text.y=element_blank(),
          #       )
  )
  # theme(axis.text.x = element_blank(), axis.ticks = element_blank()))
  dev.off()
  
  jpeg(file=paste0(res_folder,Sys.Date(),"_ABRS_BoxPlot_G100_",names(spatial_list)[ids],".jpeg"),width=960,height=960,quality = 100)
  print(ggplot(DATASET, aes(x=G100, y=ABRS, G100)) +
          geom_boxplot(width=0.5, alpha = 1, outlier.shape=NA, fill = c("red", "blue")) +
          xlab("") +
          ylab("") +
          labs(title = NULL) +
          theme(legend.position = "none", panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                panel.background = element_blank(),axis.line.x =element_line(colour = "black"),axis.line.y = element_line(colour = "black"),
                axis.text = element_text(size = rel(1.2)),
                axis.text.x = element_text(angle = 45, hjust=1, colour='black'))
        # theme(axis.title.x=element_blank(),
        #       axis.text.x=element_blank(),
        #       # axis.ticks.x=element_blank(),
        #       axis.title.y=element_blank(),
        #       axis.text.y=element_blank(),
        #       )
  )
  # theme(axis.text.x = element_blank(), axis.ticks = element_blank()))
  dev.off()
  
  res <- wilcox.test(ABRS~ G100,
                     data = DATASET,
                     exact = FALSE)
  print(paste0(names(spatial_list)[ids], ":"))
  print(res) 
  # D2: W = 7007.5, p-value = 8.396e-07. E4: W = 7052.5, p-value = 1.045e-07. 
  # E6: W = 6543.5, p-value = 0.0001499. E7: W = 5555, p-value = 0.05778.
}

# Differential expression
spatial_list_DEA <- lapply(c(1:length(spatial_list)),function(y){
  x <- spatial_list[[y]]
  MAST <- FindMarkers(x,group.by = paste0("G",n_patch),ident.1 = 'High',ident.2 = 'Low',test.use = "MAST", logfc.threshold=0)
})

spatial_list_DEA <- list(D2 = spatial_list_DEA[[1]], E4 = spatial_list_DEA[[2]], 
                         E6 = spatial_list_DEA[[3]], E7 = spatial_list_DEA[[4]])

# Volcano plot
library(ggrepel)
for (ids in c(1:length(spatial_list))) {
  genes <- geneS[geneS %in% rownames(spatial_list[[ids]][["SCT"]]@data)] # drop missing gene(s)
  
  diff_exp <- spatial_list_DEA[[ids]]
  diff_exp <- diff_exp[,c("avg_log2FC","p_val_adj")]
  write.csv(diff_exp, file=paste0(res_folder_2, '/',Sys.Date(),'_MAST_', names(spatial_list)[ids], '.csv'))
  
  diff_exp<- diff_exp %>%
    mutate(TH= case_when(diff_exp$avg_log2FC>0 & diff_exp$p_val_adj<0.05 ~"Significantly UP",
                         diff_exp$avg_log2FC< 0 & diff_exp$p_val_adj<0.05 ~"Significantly DOWN",
                         diff_exp$p_val_adj>0.05 | diff_exp$avg_log2FC==0 ~"no DEGs"))
  
  table(diff_exp$TH) 
  
  diff_exp$lab= NULL
  
  selectGenes = genes[genes %in% rownames(diff_exp)]
  print(selectGenes)
  
  # highlight ABRS genes + 15 up genes + 15 down genes
  selectGenes = c(selectGenes, rownames(na.omit(diff_exp[order(diff_exp[,'p_val_adj'],decreasing=FALSE),][diff_exp$TH=="Significantly UP",][1:15,])),
                  rownames(na.omit(diff_exp[order(diff_exp[,'p_val_adj'],decreasing=FALSE),][diff_exp$TH=="Significantly DOWN",][1:15,])))

  
  diff_exp[match(selectGenes,rownames(diff_exp)),"lab" ]= selectGenes


  ## Plot Volcano plot
  print(paste0(res_folder_2,"/",Sys.Date(),"_Volcano_Hot_Cold_",names(spatial_list)[ids],".png"))
  # png(paste0(res_folder_2,"/",Sys.Date(),"_Volcano_Hot_Cold_",names(spatial_list)[ids],"_notext.png"),width = 2000, height = 2000, res=300)
  pdf(paste0(res_folder_2,"/",Sys.Date(),"_Volcano_Hot_Cold_",names(spatial_list)[ids],".pdf"))
  print(ggplot(diff_exp,aes(x=avg_log2FC, y=-log10(p_val_adj))) +
          #geom_point(aes(colour=TH,shape=results_Data),alpha=0.5,size=4) +
          geom_point(aes(x=avg_log2FC, y=-log10(p_val_adj), colour=TH),size=3,alpha=0.5) +
          #xlab("Average Coefficients") +
          xlab("log2 Fold-Change") + # comment out for editable plot
          ylab("-log10(adjusted P-Value)") + # comment out for editable plot
          #scale_x_continuous(limits = c(-3,3))+
          theme(legend.position = "bottom",
                legend.title = element_blank(),
                legend.text =element_text(size = 15),
                axis.title = element_text(size = rel(2)),
                axis.text =element_text(size = rel(1.5)))+
          geom_hline(yintercept = -log10(0.1),lty=2,col="grey",alpha=0.5)+
          geom_vline(xintercept = 0,col="black",alpha=0.5)+
          #geom_vline(xintercept = -0.5,col="grey",alpha=0.5,lty=3)+
          #geom_vline(xintercept = 0.5,col="grey",alpha=0.5,lty=3)+
          geom_text_repel(aes(label = lab), hjust=0,nudge_x = 0.06,
                          max.overlaps=Inf, min.segment.length=0) +
          #   geom_vline(xintercept = 0.5,lty=2,col="black",alpha=0.5)+
          #   geom_vline(xintercept = -0.5,lty=2,col="black",alpha=0.5)+
          theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                panel.background = element_blank(),axis.line.x =  element_line(colour = "black"),legend.key=element_blank())+
          scale_color_manual(values=c("#999999","seagreen3","indianred"),labels=c("Not significant", "Down-regulated", "Up-regulated"))+
          guides(color=guide_legend(override.aes=list(fill=NA))))
  # theme(axis.title.x=element_blank(),
  #       axis.text.x=element_blank(),
  #       # axis.ticks.x=element_blank(),
  #       axis.title.y=element_blank(),
  #       axis.text.y=element_blank(),
  #       legend.position = "none"))
  dev.off()
  
  # Plot heatmap for selected markers (ABRS genes + 15 up genes + 15 down genes)
  if(length(selectGenes)==0) next
  fig <- lapply(selectGenes, function(gene){
    SpatialPlot(spatial_list[[ids]], features=gene)
    # ggplot2::scale_fill_gradientn(colors=colors, values=scales::rescale(c(-1,0,1))) &
    # ggplot2::scale_fill_gradientn(colors=brewer.pal(11,"Spectral")) &
    # ggplot2:: scale_fill_distiller(palette = "Spectral") &
    # theme(legend.key.size = unit(3, 'cm'), #change legend key size
    #       legend.key.height = unit(3, 'cm'), #change legend key height
    #       legend.key.width = unit(3, 'cm'), #change legend key width
    #       legend.title = element_text(size=30), #change legend title font size
    #       legend.text = element_text(size=30)) #change legend text font size
  })
  jpeg(file=paste0(res_folder,Sys.Date(),"_",names(spatial_list)[ids], "_genes.jpeg"),width=2240,height=1280,quality = 100)
  grid.arrange(grobs=fig, ncol=7)
  dev.off()
  
  # Plot heatmap for selected markers individually
  for (gene in selectGenes) {
    jpeg(file=paste0(res_folder,Sys.Date(),"_",names(spatial_list)[ids],"_",gene, ".jpeg"),width=960,height=960,quality = 100)
    print(SpatialPlot(spatial_list[[ids]], features=gene) &
            ggplot2::scale_fill_gradientn(colors=rev(brewer.pal(11,"RdBu"))) & # same as in python.
            theme(legend.key.size = unit(3, 'cm'), #change legend key size
                  legend.key.height = unit(3, 'cm'), #change legend key height
                  legend.key.width = unit(3, 'cm'), #change legend key width
                  legend.title = element_text(size=30), #change legend title font size
                  legend.text = element_text(size=30))) #change legend text font size)
    dev.off()
  }
}

# separate MAST up and dn significant genes =====================================================
files= grep("MAST_", list.files(res_folder_2), value = T)
for(ids in c(1:length(spatial_list))){
  tmp = read.csv(file.path(res_folder_2,files[ids]))
  tmp =  tmp[order(tmp[,3],decreasing=FALSE),]
  write.table(tmp[which(tmp$p_val_adj<0.05 & tmp$avg_log2FC>0),], file=file.path(res_folder_2,paste0("MAST_up_",names(spatial_list)[ids],".csv")), quote=FALSE, row.names=FALSE, sep="\t")
  write.table(tmp[which(tmp$p_val_adj<0.05 & tmp$avg_log2FC<0),], file=file.path(res_folder_2,paste0("MAST_dn_",names(spatial_list)[ids],".csv")), quote=FALSE, row.names=FALSE, sep="\t")
}
