#documentation on the functions defined hereafter

#affinity.ODE.1ligand:
#  Goal:                  Define ODEs to model the kinetic behavior of the system
#Arguments:
#- time_point:          time at which the system is evaluated   
#- state:               current values for the state variables (A,L and C)
#- parameters:          parameter values (ie, k1 and k2)

#calc_eq:
#  Goal:                  Calculate the equilibrium values for our state variables given a set of initial conditions
#Arguments:
#- A_ini:               initial value of A  
#- L_ini:               initial value of L
#- pars2:               parameter values (ie, k1 and k2)
#- returnall:           True / False, indicate whether you want the entire time-course for all state variables or just the last value of the concentration of C

#predict_Ceq:
#  Goal:                  predict the value of C at equilibrium based on the initial conditions of A and L as well as k1 and k2
#Arguments:  
#- params:              parameter values, the first of which should be k2
#- A_init:              Initial value of A
#- k1:                  k1 value
#- L_init:              Initial value of L

#residual function2:
#  Goal:                  Calculate the deviance of MFI observations to the predicted MFI from the full model 
#Arguments:  
#- params:              parameter values (ie, k2 and SF)
#- A_init_WT:           initial value of A for the WT
#- A_init_KO:           initial value of A for the KO
#- MFI_obs_KO:          a series of observed MFI values for the KO
#- MFI_obs_WT:          a series of observed MFI values for the WT
#- k1:                  k1 value
#- MFImax_WT:           maximum MFI from binding curve of WT
#- MFImax_KO:           maximum MFI from binding curve of KO

#residual function3:
#  Goal:                  Calculate the deviance of MFI observations to the predicted MFI from the reduced model 
#Arguments:  
#- params:              parameter values (ie, k2 and SF)
#- A_init_WT:           initial value of A for the WT
#- A_init_KO:           initial value of A for the KO
#- MFI_obs_KO:          a series of observed MFI values for the KO
#- MFI_obs_WT:          a series of observed MFI values for the WT
#- k1:                  k1 value
#- MFImax_WT:           maximum MFI from binding curve of WT
#- MFImax_KO:           maximum MFI from binding curve of KO

#read_MFI_data:
#  Goal:                  Read in the data for a specific cell line and a specific condition from the source document
#Arguments:  
#- xlsx_file:           path to the xlsx file in which the data is stored
#- sheet:               name of the sheet of the data of interest
#- conc_col:            name of the column  in which the concentrations are stored
#- sec_ctrl:            name of the column in which the background signal is stored
#- molar_weight_AB:     molar weigth of the antibody of interest

#determine_max:
#  Goal:                  Determine the maximum MFI for a certain titration binding curve
#Arguments:  
#- MFI_df:              dataframe as read in from read_MFI_data
#- k2_gues:             initial guess for k2
#- SF_guess:            initial guess for the scaling factor (SF)
#- label:               a label describing your dataframe used for storing the file (eg, 'B16_WT')

#compare_models:
#  Goal:                  Fit a full model and a reduced model (with and without separate k2s for the WT and KO), create figures that allow for comparison of model fits
#Arguments:  
#- WT_df:               dataframe for the WT
#- KO_df:               dataframe for the KO
#- k2_gues:             initial guess for k2
#- SF_guess:            initial guess for SF
#- label_WT:            a label describing the data used (eg, 'B16_WT')
#- label_KO:            a label describing the data used (eg, 'B16_KO')
#- celltype:            the celltype evaluated (eg, 'B16')

#oneligand_fit_list:
#  Goal:                  Fit multiple artificially created dataframes (with both WT and KO data) stored in a list using the full model to obtain parameter estimates        
#Arguments
#- WT_list              list of dataframes for the artificial WT data
#- KO_list              list of dataframes for the artificial WT data
#- k2_guess             initial guess for k2
#- SF_guess             initial guess for SF
#- label_WT             a label describing the data used (eg, 'B16_WT')
#- label_KO             a label describing the data used (eg, 'B16_KO')
#- celltype             a label that keeps track of the artificial dataframe (eg 'art0')

#singlemodelFit:
#  Goal:                  Fit the full model for a data pertaining to a single genotype
#Arguments:           
#- MFI_df               dataframe containing titration binding data
#- k2_guess             initial guess for k2
#- SF_guess             initial guess for SF
#- celltype             the celltype evaluated (eg, 'B16')


#plot_paired_metric
#  Goal:                  Show the distribution of paramater estimates from artificial data based on WT and KO simulations and compare to known theoretical values        
#Arguments:
#- data                 dataframe containing parameter estimates across artificial dataframes 
#- value_col            column relating to the parameter of interest
#- ylab                 y label axis title
#- ref_lineKO           reference line for the KO
#- ref_lineWT           reference line for the WT
#- log_scale            boolean, indicates whether the y-axis should be log-scaled

#plot_single_metric     
#  Goal:                  Show the distribution of a paramater estimate from artificial data and compare to a known theoretical value
#Arguments:
#- data                 dataframe containing parameter estimates across artificial dataframes
#- metric               column relating to the parameter of interest
#- ref_line             reference line
#- log_scale            boolean, indicates whether the y-axis should be log-scaled

#create_art_df    
#  Goal:                  Creates an artificial dataframe for a given set of parameters and constants to simulate variance
#Arguments:
#- SF                   scaling factor
#- varconstant_within   constant that determines variance within a replicate
#- varconstant_between  constant that determines variance between replicates
#- nconc                number of concentrations to simulate
#- concmax              highest concentration to simulate
#- k1                   k1 / association constant
#- k2                   k2 / dissociation constant
#- L_init               initial concentration of ligand
#- spacing              determines how concentrations are spaced in artificial data
#- extraconc            simulates addition of an extra concentration
#- nrep                 number of replicates to simulate

#create_art_df_list    
#  Goal:                  Creates a list of artificial dataframes for a given set of parameters and constants to simulate variance
#Arguments:
#- n                    number of artificial dataframes to produce
#- SF                   scaling factor
#- varconstant_within   constant that determines variance within a replicate
#- varconstant_between  constant that determines variance between replicates
#- nconc                number of concentrations to simulate
#- concmax              highest concentration to simulate
#- k1                   k1 / association constant
#- k2                   k2 / dissociation constant
#- L_init               initial concentration of ligand
#- spacing              determines how concentrations are spaced in artificial data
#- extraconc            simulates addition of an extra concentration

#check_maxbias    
#  Goal:                  Estimate the MFImax using the MM-like formula on the titration binding curve, when plot = T, the estimated max of artificial data can be compared to the theoretical maximum
#Arguments:
#- SF                   scaling factor
#- varconstant_within   constant that determines variance within a replicate
#- varconstant_between  constant that determines variance between replicates
#- k1                   k1 / association constant
#- k2                   k2 / dissociation constant
#- L_init               initial concentration of ligand
#- nrep                 number of replicates to simulate
#- top5                 determines whether the MM-llike fit is performed on all concentrations or only the highest 5 tested concentrations
#- plot                 boolean, determines whether the results are plotted


affinity.ODE.1ligand = function(time_point, state , parameters) {
  with(as.list(c(state, parameters)), {
    dA = -k1*A*L + k2*C
    dL = -k1*A*L + k2*C
    dC = k1*A*L - k2*C
    list(c(dA, dL, dC))
  })
}

calc_eq = function(A_ini, L_ini, pars2, returnall = F){
  inistate.2 = c(A=A_ini, L=L_ini, C=0)
  out.2 = ode(y=inistate.2, times=seq(0, 0.05,length.out = 100), func=affinity.ODE.1ligand, parms=pars2)
  if(returnall){
    return(as.data.frame(out.2))
  }else{
    return(tail(as.data.frame(out.2)$C, n = 1))
  }
}

predict_Ceq = function(params, A_init, k1, L_init = 1) {
  k2 = unname(params[1])         
  pars2 = c(k1 = k1, k2 = k2)
  Ceq_pred = numeric(length(A_init))
  for(i in 1:length(A_init)){
    Ceq_pred[i] = calc_eq(A_init[i], L_init, pars2)
  }
  return(Ceq_pred)
}

residual_function1 = function(params, A_init, MFI_obs, k1, MFImax) {
  k2 = unname(params[1])
  SF = unname(params[2])
  L_init = unname(MFImax / SF)
  pars = c(k1 = k1, k2 = k2)
  
  MFI_pred = numeric(length(A_init))
  for(i in 1:length(A_init)){
    MFI_pred[i] = calc_eq(A_init[i], L_init, pars) * SF
  }
  
  return(c(MFI_obs - MFI_pred))
}

residual_function2 = function(params, A_init_WT, A_init_KO, MFI_obs_WT, MFI_obs_KO, k1, MFImax_WT, MFImax_KO) {
  k2_WT = unname(params[1])
  k2_KO = unname(params[2])
  SF = unname(params[3])
  
  L_init_WT = unname(MFImax_WT / SF)
  L_init_KO = unname(MFImax_KO / SF)
  
  pars_WT = c(k1 = k1, k2 = k2_WT)
  pars_KO = c(k1 = k1, k2 = k2_KO)
  
  MFI_pred_WT = numeric(length(A_init_WT))
  MFI_pred_KO = numeric(length(A_init_KO))
  for(i in 1:length(A_init_WT)){
    MFI_pred_WT[i] = calc_eq(A_init_WT[i], L_init_WT, pars_WT) * SF
  }
  for(i in 1:length(A_init_KO)){
    MFI_pred_KO[i] = calc_eq(A_init_KO[i], L_init_KO, pars_KO) * SF
  }
  return(c(MFI_obs_WT - MFI_pred_WT, MFI_obs_KO - MFI_pred_KO))
}

residual_function3 = function(params, A_init_WT, A_init_KO, MFI_obs_WT, MFI_obs_KO, k1, MFImax_WT, MFImax_KO) {
  k2 = unname(params[1])
  SF = unname(params[2])
  
  L_init_WT = unname(MFImax_WT / SF)
  L_init_KO = unname(MFImax_KO / SF)
  
  pars = c(k1 = k1, k2 = k2)
  
  MFI_pred_WT = numeric(length(A_init_WT))
  MFI_pred_KO = numeric(length(A_init_KO))
  for(i in 1:length(A_init_WT)){
    MFI_pred_WT[i] = calc_eq(A_init_WT[i], L_init_WT, pars) * SF
  }
  for(i in 1:length(A_init_KO)){
    MFI_pred_KO[i] = calc_eq(A_init_KO[i], L_init_KO, pars) * SF
  }
  return(c(MFI_obs_WT - MFI_pred_WT, MFI_obs_KO - MFI_pred_KO))
}

read_MFI_data = function(xlsx_file, sheet, conc_col = 'Staining [ug/mL conc AB]', sec_ctrl = 'Only secondary antibody (control)', molar_weight_AB){
  
  MFI_df = as.data.frame(readxl::read_excel(xlsx_file, sheet = sheet))
  
  #Subtract the background signal
  row_offset = sum(is.na(as.numeric(MFI_df[[conc_col]])))
  col_offset = length(grep('MFI', colnames(MFI_df), invert = T))
  ctrl_row = which(MFI_df[[conc_col]] == sec_ctrl)
  
  #Select only the MFI data
  cropdf = matrix(NA, nrow(MFI_df)  -row_offset, ncol(MFI_df) -col_offset)
  for(i in 1:nrow(cropdf)){
    for(j in 1:ncol(cropdf)){
      if(!is.na(MFI_df[i + row_offset, j + col_offset])){
        cropdf[i,j] = MFI_df[i + row_offset, j + col_offset] - MFI_df[ctrl_row, j + col_offset]
      }
    }
  }
  
  cropdf = as.data.frame(cropdf)
  cropdf = cbind(MFI_df[[conc_col]][(row_offset + 1):nrow(MFI_df)], cropdf)
  colnames(cropdf) = colnames(MFI_df[col_offset:ncol(MFI_df)])
  
  #Convert concentrations reported in ug/mL to nM
  ug_ml_conc = as.numeric(cropdf$`Staining [ug/mL conc AB]`)
  cropdf$`Staining [ug/mL conc AB]` = conc2mol(conc = ug_ml_conc, unit_conc = 'microg/ml', mol_weight = molar_weight_AB, unit_mol = 'mol/L')[[1]] * 10^9
  
  df_long = cropdf %>%
    tidyr::pivot_longer(cols = starts_with('MFI'),
                        names_to = 'replicate',
                        values_to = 'MFI')
  
  colnames(df_long)[1] = 'conc'
  df_long$conc = as.numeric(df_long$conc)
  
  df_long_complete = df_long[complete.cases(df_long),] %>%
    group_by(conc) %>%
    mutate(mean_MFI = mean(MFI),) %>%
    ungroup()
  
  return(df_long_complete)
}

determine_max = function(MFI_df, k2_guess = .1, SF_guess = 100, label, plotwidth = 4, plotheight = 3, returnplot = T){
  
  high5_conc = tail(sort(unique(MFI_df$conc)), n = 5)
  conc_5 = min(high5_conc) #The 5th highest concentration
  
  MFI_df5 = MFI_df %>% 
    filter(conc %in% high5_conc) #Select only last 5 concentrations
  
  MFI_5 = mean(MFI_df5$MFI[MFI_df5$conc == conc_5]) #The MFI corresponding to the 5th highest concentration
  
  MFI_df5$MFI = MFI_df5$MFI - MFI_5
  MFI_df5$conc = MFI_df5$conc - min(MFI_df5$conc)
  
  MFI_df5 = MFI_df5 %>%
    group_by(conc) %>%
    mutate(mean_MFI = mean(MFI)) %>%
    ungroup()
  
  MFImax_hat = max(MFI_df5$mean_MFI)
  diffs = abs(MFI_df5$mean_MFI - 0.5 * max(MFI_df5$mean_MFI))
  CE50_hat = unique(MFI_df5$conc[which(diffs == min(diffs))])
  
  model = try(nls(MFI ~ (MFImax * conc) / (CE50 + conc), data = MFI_df5, start = list(MFImax = MFImax_hat, CE50 = CE50_hat)))
  
  MFImax_out = model$m$getPars()['MFImax'] + MFI_5
  
  sqrt_seq = seq(sqrt(min(MFI_df$conc)), 
                 sqrt(max(MFI_df$conc * 2)), length.out = 200)
  conc_cont = sqrt_seq^2
  
  curve = data.frame(
    conc = conc_cont + conc_5,
    MFI = predict(model, newdata = list(conc = conc_cont)) + MFI_5)
  
  p1 = ggplot() +
    geom_line(data = curve, aes(x = conc, y = MFI), color = "#486a8b", alpha = 1, linewidth = 1.5) +
    geom_point(data = MFI_df, aes(x = conc, y = MFI), color = "#486a8b", size = 2, alpha = .5) + 
    scale_x_log10() +
    labs(
      title = sub('_', ' ', label),
      x = TeX('$\\[A_{init}\\]$ (nM)'),
      y = 'MFI'
    ) + 
    annotate("text", 
             x = 1, 
             y = max(curve$MFI), 
             label = bquote(~ MFI[max] ~ "=" ~ .(round(MFImax_out))), 
             hjust = 0, vjust = 1, size = 4, color = "#486a8b") +
    theme_classic() + 
    theme(panel.grid = element_blank())
  
  if(returnplot){
    print(p1)
  }
  
  ggsave(paste('../Figures/Supporting_Figures/Max_estimation_', label, '.png', sep = ''), plot = p1, width = plotwidth, height = plotheight, dpi = 500)
  return(MFImax_out)
}

compare_models = function(WT_df, KO_df, k2_guess = .2, SF_guess = 100, label_WT, label_KO, celltype){
  
  initial_guesses = c(k2_WT =  k2_guess, k2_KO = k2_guess, SF = SF_guess)
  
  WT_max = determine_max(WT_df, label = label_WT)
  KO_max = determine_max(KO_df, label = label_KO)
  
  #Fit the model assuming shared SF and separate k2
  fit2 = nls.lm(par = initial_guesses, fn = residual_function2, 
                A_init_WT = WT_df$conc, A_init_KO = KO_df$conc, MFI_obs_WT = WT_df$MFI, MFI_obs_KO = KO_df$MFI, k1 = 1, MFImax_WT = WT_max, MFImax_KO = KO_max, lower = c(0,0,0))
  
  k2_WT_estimated = coef(fit2)['k2_WT']
  k2_KO_estimated = coef(fit2)['k2_KO']
  SF_estimated = coef(fit2)['SF']
  
  
  #Generate the data for plotting
  A_ini_WT = unname(max(WT_df$conc))
  A_ini_KO = unname(max(KO_df$conc))
  L_ini_WT = unname(WT_max / SF_estimated)
  L_ini_KO = unname(KO_max / SF_estimated)
  pars_WT = list(k1 = 1, k2 = k2_WT_estimated)
  pars_KO = list(k1 = 1, k2 = k2_KO_estimated)
  
  #Compute full trajectory
  timecourseWT = calc_eq(A_ini_WT, L_ini_WT, pars_WT, returnall = TRUE)
  timecourseKO = calc_eq(A_ini_KO, L_ini_KO, pars_KO, returnall = TRUE)
  
  timecourseDF = bind_rows(timecourseWT, timecourseKO, .id = 'genotype')
  
  #Reshape the data
  plot_data = timecourseDF %>%
    pivot_longer(cols = c(A, L, C), names_to = "variable", values_to = "value")
  
  plot_data$variable = factor(plot_data$variable, levels = c('A','L','C'))
  plot_data$genotype = factor(plot_data$genotype)
  levels(plot_data$genotype) = c('WT', 'KO')
  
  p2.1 = ggplot(plot_data, aes(x = time, y = value, color = genotype)) +
    geom_line(linewidth = 1) +  #Set linewidth to 1
    scale_color_manual(values = c("WT" = "#486a8b", "KO" = "#a55d5d"), name = "Genotype") +  
    facet_grid(variable ~ ., scales = "free_y") +  #Facet by variable
    theme_minimal() +
    labs(title = "Time Courses for A, L, and C", x = "Time (a.u.)", y = "Concentration (nM)") +
    theme_bw() + 
    theme(panel.grid = element_blank())
  
  ggsave(paste('../Figures/Supporting_figures/Timecourse_', celltype, '_fullmodel.png', sep = ''), plot = p2.1, width = 4, height = 3, dpi = 500)
  
  
  conc = seq(0, max(c(WT_df$conc, KO_df$conc)), length.out = 500)
  MFIpred_WT = predict_Ceq(c(k2 = k2_WT_estimated), conc, k1 = 1, L_init = unname(WT_max / SF_estimated)) * SF_estimated
  MFIpred_KO = predict_Ceq(c(k2 = k2_KO_estimated), conc, k1 = 1, L_init = unname(KO_max / SF_estimated)) * SF_estimated
  
  plot_data = data.frame(conc, MFIpred_WT, MFIpred_KO)
  
  plot_data = plot_data %>%
    pivot_longer(cols = c('MFIpred_WT', 'MFIpred_KO'), names_to = 'genotype', values_to = 'MFI')
  
  plot_data$genotype = sub('MFIpred_', '', plot_data$genotype)
  
  AIC_full = round(length(fit2$fvec) * log(sum(fit2$fvec^2) / length(fit2$fvec)) + 2 * length(fit2$par), digits = 1)
  
  #Plot the data
  p3.1 = ggplot(plot_data, aes(x = conc, y = MFI, group = genotype, color = genotype)) +
    geom_line(linewidth = 1.5) +
    scale_color_manual(values = c("WT" = "#486a8b", "KO" = "#a55d5d")) +  
    geom_point(data = WT_df, aes(x = conc, y = MFI), size = 2, color = '#486a8b', alpha = .5, inherit.aes = FALSE) + 
    geom_point(data = KO_df, aes(x = conc, y = MFI), size = 2, color = '#a55d5d', alpha = .5, inherit.aes = FALSE) + 
    scale_x_log10() +
    theme_classic() +
    theme(panel.grid = element_blank(), legend.position = 'right') + 
    labs(
      title = paste(celltype, 'AIC = ', AIC_full),
      x = TeX('$\\[A_{init}\\]$ (nM)'), 
      y = "MFI",
      color = "Genotype"
    ) +
    annotate("text", x = min(plot_data$conc), y = max(plot_data$MFI), 
             label = bquote("WT " ~ K[D] ~ "=" ~ .(k2_WT_estimated)), 
             color = "#486a8b", hjust = 0, size = 4) +
    annotate("text", x = min(plot_data$conc), y = max(plot_data$MFI) * 0.9, 
             label = bquote("KO " ~ K[D] ~ "=" ~ .(k2_KO_estimated)), 
             color = "#a55d5d", hjust = 0, size = 4)
  
  print(p3.1)
  ggsave(paste('../Figures/Main_figures/Fitted_curves_fullmodel', celltype, '.png', sep = ''), plot = p3.1, width = 4, height = 3, dpi = 500)
  
  ########################Start again with reduced model########################
  
  #Fit the model assuming shared SF and shared k2
  initial_guesses = c(k2 =  k2_guess, SF = SF_guess)
  fit3 = nls.lm(par = initial_guesses, fn = residual_function3, 
                A_init_WT = WT_df$conc, A_init_KO = KO_df$conc, MFI_obs_WT = WT_df$MFI, MFI_obs_KO = KO_df$MFI, k1 = 1, MFImax_WT = WT_max, MFImax_KO = KO_max, lower = c(0,0))
  
  k2_estimated = coef(fit3)['k2']
  SF_estimated = coef(fit3)['SF']
  
  #Generate the data for plotting
  A_ini_WT = unname(max(WT_df$conc))
  A_ini_KO = unname(max(KO_df$conc))
  L_ini_WT = unname(WT_max / SF_estimated)
  L_ini_KO = unname(KO_max / SF_estimated)
  pars = list(k1 = 1, k2 = k2_estimated)
  
  #Compute full trajectory
  timecourseWT = calc_eq(A_ini_WT, L_ini_WT, pars, returnall = TRUE)
  timecourseKO = calc_eq(A_ini_KO, L_ini_KO, pars, returnall = TRUE)
  
  timecourseDF = bind_rows(timecourseWT, timecourseKO, .id = 'genotype')
  
  #Reshape the data
  plot_data = timecourseDF %>%
    pivot_longer(cols = c(A, L, C), names_to = "variable", values_to = "value")
  
  plot_data$variable = factor(plot_data$variable, levels = c('A','L','C'))
  plot_data$genotype = factor(plot_data$genotype)
  levels(plot_data$genotype) = c('WT', 'KO')
  
  p2.2 = ggplot(plot_data, aes(x = time, y = value, color = genotype)) +
    geom_line(linewidth = 1) +  
    scale_color_manual(values = c("WT" = "#486a8b", "KO" = "#a55d5d"), name = "Genotype") +  
    facet_grid(variable ~ ., scales = "free_y") + 
    theme_minimal() +
    labs(title = "Time Courses for A, L, and C", x = "Time (a.u.)", y = "Concentration (nM)") +
    theme_bw() + 
    theme(panel.grid = element_blank())
  
  ggsave(paste('../Figures/Supporting_figures/Timecourse_', celltype, '_reducedmodel.png', sep = ''), plot = p2.2, width = 4, height = 3, dpi = 500)
  
  conc = seq(0, max(c(WT_df$conc, KO_df$conc)), length.out = 500)
  MFIpred_WT = predict_Ceq(c(k2 = k2_estimated), conc, k1 = 1, L_init = unname(WT_max / SF_estimated)) * SF_estimated
  MFIpred_KO = predict_Ceq(c(k2 = k2_estimated), conc, k1 = 1, L_init = unname(KO_max / SF_estimated)) * SF_estimated
  
  plot_data = data.frame(conc, MFIpred_WT, MFIpred_KO)
  
  plot_data = plot_data %>%
    pivot_longer(cols = c('MFIpred_WT', 'MFIpred_KO'), names_to = 'genotype', values_to = 'MFI')
  
  plot_data$genotype = sub('MFIpred_', '', plot_data$genotype)
  
  AIC_reduced = round(length(fit3$fvec) * log(sum(fit3$fvec^2) / length(fit3$fvec)) + 2 * length(fit3$par), digits = 1)
  
  #Plot the data
  p3.2 = ggplot(plot_data, aes(x = conc, y = MFI, group = genotype, color = genotype)) +
    geom_line(linewidth = 1.5) +
    scale_color_manual(values = c("WT" = "#486a8b", "KO" = "#a55d5d")) +  
    geom_point(data = WT_df, aes(x = conc, y = MFI), size = 2, color = '#486a8b', alpha = .5, inherit.aes = FALSE) + 
    geom_point(data = KO_df, aes(x = conc, y = MFI), size = 2, color = '#a55d5d', alpha = .5, inherit.aes = FALSE) + 
    scale_x_log10() +
    theme_classic() +
    theme(panel.grid = element_blank(), legend.position = 'right') +
    labs(
      title = paste(celltype, 'AIC = ', AIC_reduced),
      x = TeX('$\\[A_{init}\\]$ (nM)'), 
      y = "MFI",
      color = "Genotype"
    ) +
    annotate("text", x = min(plot_data$conc), y = max(plot_data$MFI), 
             label = bquote("WT " ~ K[D] ~ "=" ~ .(k2_estimated)), 
             color = "#486a8b", hjust = 0, size = 4) +
    annotate("text", x = min(plot_data$conc), y = max(plot_data$MFI) * 0.9, 
             label = bquote("KO " ~ K[D] ~ "=" ~ .(k2_estimated)), 
             color = "#a55d5d", hjust = 0, size = 4)
  
  print(p3.2)
  ggsave(paste('../Figures/Main_figures/Fitted_curves_reducedmodel', celltype, '.png', sep = ''), plot = p3.2, dpi = 500, width = 4, height = 3)
}

oneligand_fit_list = function(WT_list, KO_list, k2_guess = 0.2, SF_guess = 100, label_WT, label_KO, celltype, plot = F) {
  n = length(WT_list)
  results = data.frame(
    replicate = seq_len(n),
    k2_WT = numeric(n),
    k2_KO = numeric(n),
    SF = numeric(n),
    WT_max = numeric(n),
    KO_max = numeric(n),
    AIC = numeric(n),
    stringsAsFactors = FALSE
  )
  
  for(i in seq_len(n)){
    WT_df = WT_list[[i]]
    KO_df = KO_list[[i]]
    
    initial_guesses = c(k2_WT =  k2_guess, k2_KO = k2_guess, SF = SF_guess)
    
    WT_max = determine_max(WT_df, label = label_WT)
    KO_max = determine_max(KO_df, label = label_KO)
    
    fit2 = nls.lm(par = initial_guesses, fn = residual_function2, 
                  A_init_WT = WT_df$conc, A_init_KO = KO_df$conc, 
                  MFI_obs_WT = WT_df$MFI, MFI_obs_KO = KO_df$MFI, 
                  k1 = 1, MFImax_WT = WT_max, MFImax_KO = KO_max, 
                  lower = c(0, 0, 0))
    
    k2_WT_estimated = coef(fit2)['k2_WT']
    k2_KO_estimated = coef(fit2)['k2_KO']
    SF_estimated = coef(fit2)['SF']
    
    # Store results
    results$k2_WT[i] = k2_WT_estimated
    results$k2_KO[i] = k2_KO_estimated
    results$SF[i] = SF_estimated
    results$WT_max[i] = WT_max
    results$KO_max[i] = KO_max
    results$AIC[i] = round(length(fit2$fvec) * log(sum(fit2$fvec^2) / length(fit2$fvec)) + 2 * length(fit2$par), digits = 1)
    
    if(plot){
      
      # Generate predictions
      conc = seq(0, max(c(WT_df$conc, KO_df$conc)), length.out = 500)
      MFIpred_WT = predict_Ceq(c(k2 = k2_WT_estimated), conc, k1 = 1, L_init = unname(WT_max / SF_estimated)) * SF_estimated
      MFIpred_KO = predict_Ceq(c(k2 = k2_KO_estimated), conc, k1 = 1, L_init = unname(KO_max / SF_estimated)) * SF_estimated
      
      plot_data = data.frame(conc, MFIpred_WT, MFIpred_KO) %>%
        pivot_longer(cols = c('MFIpred_WT', 'MFIpred_KO'), names_to = 'genotype', values_to = 'MFI') %>%
        mutate(genotype = sub('MFIpred_', '', genotype))
      
      p3.1 = ggplot(plot_data, aes(x = conc, y = MFI, group = genotype, color = genotype)) +
        geom_line(linewidth = 1) +
        scale_color_manual(values = c("WT" = "#486a8b", "KO" = "#a55d5d")) +  
        geom_point(data = WT_df, aes(x = conc, y = MFI), size = 2, color = '#486a8b', alpha = .5, inherit.aes = FALSE) + 
        geom_point(data = KO_df, aes(x = conc, y = MFI), size = 2, color = '#a55d5d', alpha = .5, inherit.aes = FALSE) + 
        scale_x_log10() +
        theme_classic() +
        theme(panel.grid = element_blank(), legend.position = 'right') +
        labs(
          title = paste(celltype, '- replicate', i, 'AIC =', results$AIC[i]),
          x = TeX('$\\[A_{init}\\]$ (nM)'), 
          y = "MFI",
          color = "Genotype"
        ) +
        annotate("text", x = min(plot_data$conc), y = max(plot_data$MFI),
                 label = bquote("WT " ~ K[D] ~ "=" ~ .(round(k2_WT_estimated, 2))), 
                 color = "#486a8b", hjust = 0, size = 4) +
        annotate("text", x = min(plot_data$conc), y = max(plot_data$MFI) * 0.9,
                 label = bquote("KO " ~ K[D] ~ "=" ~ .(round(k2_KO_estimated, 2))), 
                 color = "#a55d5d", hjust = 0, size = 4)
      
      ggsave(paste0('../Figures/Supporting_figures/Fitted_curves_fullmodel_', celltype, '_rep', i, '.png'), width = 4, height = 3, plot = p3.1, dpi = 500)
    }
  }
  return(results)
}

singlemodelFit = function(MFI_df, k2_guess = .2, SF_guess = 100, celltype){
  
  initial_guesses = c(k2 =  k2_guess, SF = SF_guess)
  MFI_max = determine_max(MFI_df, label = celltype)
  
  #Fit the model assuming shared SF and separate k2
  fit1 = nls.lm(par = initial_guesses, fn = residual_function1, 
                A_init = MFI_df$conc, MFI_obs = MFI_df$MFI, k1 = 1, MFImax = MFI_max, lower = c(0,0))
  
  k2_estimated = coef(fit1)['k2']
  SF_estimated = coef(fit1)['SF']
  
  #Generate the data for plotting
  A_ini = unname(max(MFI_df$conc))
  L_ini = unname(MFI_max / SF_estimated)
  pars = list(k1 = 1, k2 = k2_estimated)
  
  #Compute full trajectory
  timecourse = calc_eq(A_ini, L_ini, pars, returnall = TRUE)
  
  #Reshape the data
  plot_data = timecourse %>%
    pivot_longer(cols = c(A, L, C), names_to = "variable", values_to = "value")
  plot_data$variable = factor(plot_data$variable, levels = c('A','L','C'))
  
  p2.1 = ggplot(plot_data, aes(x = time, y = value)) +
    geom_line(linewidth = 1, col = "#486a8b") +  #Set linewidth to 1
    #scale_color_manual(values = c("MFI" = "#486a8b", "KO" = "#a55d5d"), name = "Genotype") +  
    facet_grid(variable ~ ., scales = "free_y") +  #Facet by variable
    labs(title = "Time Courses for A, L, and C", x = "Time (a.u.)", y = "Concentration (nM)") +
    theme_bw() + 
    theme(panel.grid = element_blank())
  
  ggsave(paste('../Figures/Main_figures/Timecourse_', celltype, '_1genotype.png', sep = ''), width = 4, height = 3, plot = p2.1, dpi = 500)
  
  conc = seq(0, max(MFI_df$conc), length.out = 500)
  pred_MFI = predict_Ceq(c(k2 = k2_estimated), conc, k1 = 1, L_init = unname(MFI_max / SF_estimated)) * SF_estimated
  
  plot_data = data.frame(conc, pred_MFI)
  colnames(plot_data)[2] = 'MFI'
  
  RMSE = round(sqrt(mean(fit1$fvec^2)), digits = 1)
  
  #Plot the data
  p3.1 = ggplot(plot_data, aes(x = conc, y = MFI)) +
    geom_line(linewidth = 1.5, col = "#486a8b") +
    geom_point(data = MFI_df, aes(x = conc, y = MFI), size = 2, color = '#486a8b', alpha = .5, inherit.aes = FALSE) + 
    scale_x_log10() +
    theme_classic() +
    theme(panel.grid = element_blank(), legend.position = 'right',
          #plot.title = element_text(size = 20),       
          #axis.title = element_text(size = 16),          
          #axis.text = element_text(size = 14),                          
          #legend.title = element_text(size = 16),        
          #legend.text = element_text(size = 14)                         
    ) +  #Enable legend
    labs(
      title = 'Evaluate model performance',
      x = TeX('$\\[A_{init}\\]$ (nM)'), 
      y = "MFI",
      color = "Genotype"
    ) +
    annotate("text", x = min(plot_data$conc), y = max(plot_data$MFI), 
             label = bquote(~ RMSE ~ "=" ~ .(round(RMSE,1))), 
             color = "#486a8b", hjust = 0, size = 4)
  
  print(p3.1)
  ggsave(paste('../Figures/Main_figures/Fitted_curves_1genotype', celltype, '.png', sep = ''), plot = p3.1, dpi = 500, width = 4, height = 3)
}

plot_paired_metric = function(data, value_col, ylab = "", ref_lineWT = NULL, ref_lineKO = NULL, log_scale = FALSE) {
  summary_df = data %>%
    group_by(pair, genotype) %>%
    summarise(
      median = median(.data[[value_col]]),
      sd = sd(.data[[value_col]]),
      .groups = "drop"
    )
  
  p = ggplot(data, aes(x = pair, y = .data[[value_col]], fill = genotype)) +
    stat_summary(
      aes(group = genotype),
      fun = median,
      geom = "bar",
      color = 'black',
      size = 1,
      position = 'dodge',
      width = 0.8
    ) +
    
    {if (!is.null(ref_lineKO)) geom_hline(yintercept = ref_lineWT, linetype = "dashed", color = "#486a8b", linewidth = 1)} +
    {if (!is.null(ref_lineKO)) geom_hline(yintercept = ref_lineKO, linetype = "dashed", color = "#a55d5d", linewidth = 1)} +
    
    geom_errorbar(
      data = summary_df,
      aes(x = pair, y = median, ymin = median - sd, ymax = median + sd, group = genotype),
      position = position_dodge(width = 0.8),
      width = 0.4,
      size = 1,
      inherit.aes = FALSE
    ) +
    geom_quasirandom(
      aes(group = genotype),
      dodge.width = 0.8,
      width = 0.2,
      size = 1,
      alpha = 0.5,
      show.legend = F
    ) +
    scale_fill_manual(values = c("WT" = "#486a8b", "KO" = "#a55d5d")) +
    scale_color_manual(values = c("WT" = "#486a8b", "KO" = "#a55d5d")) +
    theme_classic() +
    labs(x = "Artificial dataset", y = ylab)
  
  
  if (log_scale) {
    p = p + scale_y_log10(expand = c(0, 0))
  }else{
    p = p + scale_y_continuous(expand = c(0, 0))
  }
  
  p = p + coord_cartesian(clip = 'off')
  
  print(p)
  ggsave(paste0('../Figures/Supporting_figures/accuracy_', value_col, '_artdata.png'), plot = p, dpi = 500, width = 4, height = 3)
}

plot_single_metric = function(data, metric, ref_line = NULL, log_scale = TRUE) {
  summary_df = data %>%
    group_by(source) %>%
    summarise(
      median = median(.data[[metric]]),
      sd = sd(.data[[metric]]),
      .groups = "drop"
    )
  
  p = ggplot(data, aes(x = source, y = .data[[metric]])) +
    stat_summary(fun = median, 
                 geom = "bar", 
                 fill = "#486a8b", 
                 color = "black", 
                 size = 1,
                 width = 0.6) +
    
    # ref line first: bottom layer
    {if (!is.null(ref_line)) geom_hline(yintercept = ref_line, linetype = "dashed", color = "red", linewidth = 1)} +
    
    # error bars next
    geom_errorbar(data = summary_df,
                  aes(x = source, y = median, ymin = median - sd, ymax = median + sd),
                  width = 0.2, linewidth = 1, inherit.aes = FALSE) +
    
    # data points last: top layer
    geom_quasirandom(width = 0.3, size = 2, alpha = 0.5) +
    
    theme_classic() +
    labs(x = "Artificial Dataset", y = paste(metric, 'estimates'))
  
  if (log_scale) {
    p = p + scale_y_log10(expand = c(0, 0))
  } else {
    p = p + scale_y_continuous(expand = c(0, 0))
  }
  
  p = p + coord_cartesian(clip = 'off')
  
  print(p)
  ggsave(paste0('../Figures/Supporting_figures/accuracy_', metric, '_artdata.png'), plot = p, dpi = 500, width = 4, height = 3)
}

create_art_df = function(SF, varconstant_within, varconstant_between, nconc = 9, concmax= 1500, k1, k2, L_init = 10, spacing = 'log', extraconc = F, nrep = 3){
  
  conc = numeric(nconc)
  
  if(spacing == 'log'){
    for(i in 0:(nconc - 1)){
      conc[i + 1] = concmax / 2^i
    }
  }else if(spacing == 'linear'){
    conc = seq(0, concmax, length.out = nconc + 1)[-1]
  }
  
  #add an extra concentration between the highest two concentrations
  if(extraconc){
    concextra = mean(conc[1:2])
    conc = c(conc[1], concextra, conc[-1])
    nconc = nconc + 1
  }
  
  MFI_uniq = predict_Ceq(c(k2 = k2), conc, k1 = k1, L_init = L_init) * SF
  
  conc = rep(conc, each = nrep)
  MFI = rep(MFI_uniq, each = nrep)
  
  rep_levels = c('MFI1', 'MFI2', 'MFI3')
  replicate_labels = rep(rep_levels, times = length(MFI_uniq))
  
  between_scalers = rnorm(length(rep_levels), mean = 0, 
                          sd = varconstant_between)  #not yet scaled by MFI
  
  #Map replicate label to its between-scaler
  replicate_scaler_map = setNames(between_scalers, rep_levels)
  
  #Build vector of between-replicate residuals scaled by MFI
  resids_between = sapply(seq_along(MFI), function(i) {
    replicate_label = replicate_labels[i]
    replicate_scaler_map[replicate_label] * MFI[i]
  })
  
  #Add within-replicate noise (already correct)
  resids_within = rnorm(length(MFI), 0, varconstant_within * MFI)
  
  #Final MFI with both noise sources
  MFI = MFI + resids_within + resids_between
  
  #Build final dataframe
  art_df = data.frame(conc = conc,
                      replicate = replicate_labels,
                      MFI = MFI) %>%
    group_by(conc) %>%
    mutate(mean_MFI = mean(MFI)) %>%
    ungroup()
  
  return(art_df)
}

#Wrapper function to generate a list of 10 datasets
create_art_df_list = function(n = 10, SF, varconstant_within, varconstant_between, nconc = 9, concmax = 1500, k1, k2, L_init = 10, spacing = 'log', extraconc = F) {
  out_list = vector("list", n)
  for (i in seq_len(n)) {
    out_list[[i]] = create_art_df(SF = SF, varconstant_within = varconstant_within, varconstant_between = varconstant_between, nconc = nconc, concmax = concmax,
                                  k1 = k1, k2 = k2, L_init = L_init, spacing = spacing, extraconc = extraconc)
  }
  return(out_list)
}

#Wrapper function to generate a list of 10 datasets
create_art_df_list = function(n = 10, SF, varconstant_within, varconstant_between, nconc = 9, concmax = 1500, k1, k2, L_init = 10, spacing = 'log', extraconc = F) {
  out_list = vector("list", n)
  for (i in seq_len(n)) {
    out_list[[i]] = create_art_df(SF = SF, varconstant_within = varconstant_within, varconstant_between = varconstant_between, nconc = nconc, concmax = concmax,
                                  k1 = k1, k2 = k2, L_init = L_init, spacing = spacing, extraconc = extraconc)
  }
  return(out_list)
}

check_maxbias = function(SF = 350, varconstant_within = 0, varconstant_between =0, k1 = 1, k2 = 90, L_init = 40, nrep = 1, top5 = F, plot = T){
  
  artdf = create_art_df(SF = SF, varconstant_within = varconstant_within, varconstant_between = varconstant_between, k1 = k1, k2 = k2, L_init = L_init, nrep = nrep)
  
  if(top5){
    high5_conc = tail(sort(unique(artdf$conc)), n = 5)
    conc_5 = min(high5_conc) #The 5th highest concentration
    
    artdf5 = artdf %>% 
      filter(conc %in% high5_conc) #Select only last 5 concentrations
    
    art_5 = mean(artdf5$MFI[artdf5$conc == conc_5]) #The MFI corresponding to the 5th highest concentration
    
    artdf5$MFI = artdf5$MFI - art_5
    artdf5$conc = artdf5$conc - min(artdf5$conc)
    
    artdf5 = artdf5 %>%
      group_by(conc) %>%
      mutate(mean_MFI = mean(MFI)) %>%
      ungroup()
    
    MFImax_hat = max(artdf5$mean_MFI)
    diffs = abs(artdf5$mean_MFI - 0.5 * max(artdf5$mean_MFI))
    CE50_hat = unique(artdf5$conc[which(diffs == min(diffs))])
    
    modelMM = try(nls(MFI ~ (MFImax * conc) / (CE50 + conc), data = artdf5, start = list(MFImax = MFImax_hat, CE50 = CE50_hat)))
    MFImax_MM = modelMM$m$getPars()['MFImax'] + art_5
    
    sqrt_seq = seq(sqrt(min(artdf$conc)), 
                   sqrt(max(artdf$conc * 2)), length.out = 200)
    conc_cont = sqrt_seq^2
    
    artdata = data.frame(
      conc = artdf5$conc,
      MFI = artdf5$MFI
    )
    
    curve = data.frame(
      conc = conc_cont + conc_5,
      MFI = predict(modelMM, newdata = list(conc = conc_cont)) + art_5)
  }else{
    MFImax_hat = max(artdf$mean_MFI)
    diffs = abs(artdf$mean_MFI - 0.5 * max(artdf$mean_MFI))
    CE50_hat = unique(artdf$conc[diffs == min(diffs)])
    n_hat = 1
    k_hat = .003
    
    modelMM = nls(MFI ~ (MFImax * conc) / (CE50 + conc), data = artdf, start = list(MFImax = MFImax_hat, CE50 = CE50_hat))
    MFImax_MM = modelMM$m$getPars()['MFImax']
    
    #Extract model parameters
    params = coef(modelMM)
    MFImax_MM = params["MFImax"]
    CE50_MM = params["CE50"]
    
    #Create label string for legend
    fit_label = sprintf("Fit: MFImax = %.2f", MFImax_MM)
    
    artdata = data.frame(
      conc = artdf$conc,
      MFI = artdf$MFI
    )
    
    #Generate smooth prediction data
    conc_smooth = exp(seq(log(min(artdf$conc)), log(max(artdf$conc)), length.out = 200))
    curve = data.frame(
      conc = conc_smooth,
      MFI = (MFImax_MM * conc_smooth) / (CE50_MM + conc_smooth)
    )
  }
  
  if(plot){
    outplot = plot_data = rbind(artdata, curve)
    
    legend_levels = c("Simulated data", "Fitted curve", "Predicted max", "True max")
    
    maxfitPlot = ggplot() +
      geom_hline(aes(yintercept = MFImax_MM, color = "Predicted max", linetype = "Predicted max"), linewidth = 1.5) +
      geom_hline(aes(yintercept = SF * L_init, color = "True max", linetype = "True max"), linewidth = 1.5) +
      geom_point(data = artdf, aes(x = conc, y = MFI, color = "Simulated data", shape = "Simulated data"), size = 3) +
      geom_line(data = curve, aes(x = conc, y = MFI, color = "Fitted curve", linetype = "Fitted curve"), linewidth = 1.5) +
      
      #Manual color and linetype
      scale_color_manual(
        name = NULL,
        values = c(
          "Simulated data" = "#486a8b",
          "Fitted curve" = "#486a8b",
          "Predicted max" = "#a5bbd2",
          "True max" = "red"
        ),
        breaks = legend_levels
      ) +
      scale_linetype_manual(
        name = NULL,
        values = c(
          "Simulated data" = "blank",
          "Fitted curve" = "solid",
          "Predicted max" = "solid",
          "True max" = "dashed"
        ),
        breaks = legend_levels
      ) +
      scale_shape_manual(
        name = NULL,
        values = c(
          "Simulated data" = 16,
          "Fitted curve" = NA,
          "Predicted max" = NA,
          "True max" = NA
        ),
        breaks = legend_levels
      ) +
      
      scale_x_log10() +
      labs(x = TeX("$\\[A_{init}\\]$ (a.u.)"),
           y = "MFI"
      ) +
      theme_classic() +
      
      #Fix the legend appearance
      guides(
        color = guide_legend(
          override.aes = list(
            linetype = c("blank", "solid", "solid", "dashed"),
            shape = c(16, NA, NA, NA)
          )
        ),
        linetype = "none",
        shape = "none"
      )
    
    print(maxfitPlot)
    
    if(top5){
      ggsave(paste0('../Figures/Supporting_figures/maxfitPlot_top5_KD', k2/k1, '.png'), width = 5, height = 3, dpi = 600)
    }else{
      ggsave(paste0('../Figures/Supporting_figures/maxfitPlot_fullrange_KD', k2/k1, '.png'), width = 5, height = 3, dpi = 600)
    }
    
  }else{
    return(MFImax_MM)
  }
}