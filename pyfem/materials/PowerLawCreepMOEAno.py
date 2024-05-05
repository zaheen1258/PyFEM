################################################################################
#  This Python file is part of PyFEM, the code that accompanies the book:      #
#                                                                              #
#    'Non-Linear Finite Element Analysis of Solids and Structures'             #
#    R. de Borst, M.A. Crisfield, J.J.C. Remmers and C.V. Verhoosel            #
#    John Wiley and Sons, 2012, ISBN 978-0470666449                            #
#                                                                              #
#  Copyright (C) 2011-2023. The code is written in 2011-2012 by                #
#  Joris J.C. Remmers, Clemens V. Verhoosel and Rene de Borst and since        #
#  then augmented and maintained by Joris J.C. Remmers.                        #
#  All rights reserved.                                                        #
#                                                                              #
#  A github repository, with the most up to date version of the code,          #
#  can be found here:                                                          #
#     https://github.com/jjcremmers/PyFEM/                                     #
#     https://pyfem.readthedocs.io/                                            #
#                                                                              #
#  The original code can be downloaded from the web-site:                      #
#     http://www.wiley.com/go/deborst                                          #
#                                                                              #
#  The code is open source and intended for educational and scientific         #
#  purposes only. If you use PyFEM in your research, the developers would      #
#  be grateful if you could cite the book.                                     #
#                                                                              #
#  Disclaimer:                                                                 #
#  The authors reserve all rights but do not guarantee that the code is        #
#  free from errors. Furthermore, the authors shall not be liable in any       #
#  event caused by the use of the program.                                     #
################################################################################

from pyfem.materials.BaseMaterial import BaseMaterial
from pyfem.materials.MatUtils     import vonMisesStress,hydrostaticStress
from numpy import zeros, ones, dot, array, outer
from math import sqrt
import copy
from pyfem.materials import MoEModel
import numpy as np
import math

class PowerLawCreepMOEAno( BaseMaterial ):

  def __init__ ( self, props ):

    self.tolerance = 1e-8

    BaseMaterial.__init__( self, props )

    self.rho_c = self.rhoc
    self.rho_w = self.rhow
    self.Tmp = self.Temp
    self.Flux = self.flux
    self.ebulk3  = self.E / ( 1.0 - 2.0*self.nu )
    self.eg2     = self.E / ( 1.0 + self.nu )
    self.eg      = 0.5*self.eg2
    self.eg3     = 3.0*self.eg
    self.elam    = ( self.ebulk3 - self.eg2 ) / 3.0

    self.ctang = zeros(shape=(6,6))

    self.ctang[:3,:3] = self.elam

    self.ctang[0,0] += self.eg2
    self.ctang[1,1] = self.ctang[0,0]
    self.ctang[2,2] = self.ctang[0,0]

    self.ctang[3,3] = self.eg
    self.ctang[4,4] = self.ctang[3,3]
    self.ctang[5,5] = self.ctang[3,3]

    self.setHistoryParameter( 'sigma'  , zeros(6) )
    self.setHistoryParameter( 'eelas'  , zeros(6) )
    self.setHistoryParameter( 'eplas'  , zeros(6) )
    self.setHistoryParameter( 'eqplas' , zeros(1) )
    #self.setHistoryParameter( 'old_deqpl' , zeros(1) )
    self.setHistoryParameter( 'rohc' , zeros(1) )
    self.setHistoryParameter( 'rohw' , zeros(1) )
    # self.setHistoryParameter( 'total_deqpl' , zeros(1) )

    self.commitHistory()

    #Set the labels for the output data in this material model
    self.outLabels = [ "S11" , "S22" , "S33" , "S23" , "S13" , "S12" , "Epl", "rho_c","rho_w"]
    self.outData   = zeros(9)

#------------------------------------------------------------------------------
#  pre:  kinematics object containing current strain (kinemtics.strain)
#  post: stress vector and tangent matrix
#------------------------------------------------------------------------------

  def getStress( self, kinematics):

    eelas  = self.getHistoryParameter('eelas')
    eplas  = self.getHistoryParameter('eplas')
    eqplas = self.getHistoryParameter('eqplas')
    sigma_old  = self.getHistoryParameter('sigma')
    #old_creep = self.getHistoryParameter('old_deqpl')
    rhoc_old = self.getHistoryParameter('rohc')
    rhow_old = self.getHistoryParameter('rohw')
    # ol_tot_deqpl = self.getHistoryParameter('total_deqpl')
    #eqplas = eqplas[0].astype(float)

    if eqplas == 0:
      eqplas = 1e-11

    # if ol_tot_deqpl == 0:
    #   ol_tot_deqpl = 0

    # if old_creep == 0:
    #   old_creep = 0

    if rhoc_old <= 0:
      rhoc_old = self.rho_c

    if rhow_old <= 0:
      rhow_old = self.rho_w

    if rhoc_old > 8461801410123.313:
      rhoc_old = 8461801410123.313

    if rhow_old > 11999567054170.322:
      rhow_old = 11999567054170.322

    # ol_rhoc = 1e12
    # ol_rhow = 1e11

    rhoc = copy.copy(rhoc_old)
    rhow = copy.copy(rhow_old)

    sigma = copy.copy(sigma_old)

    sigma += dot( self.ctang , kinematics.dstrain)

    tang = self.ctang
    # total_strain = vonMisesStress(eelas) + ol_tot_deqpl

    eelas += kinematics.dstrain

    trial_stress = sigma_old + dot( self.ctang , kinematics.dstrain )

    smises = vonMisesStress( trial_stress )

    #exp_time = pow(self.solverStat.time, self.m)
    #print('------------------------------')
    #print(f"von-mises trial stress: {smises}")
    #print('------------------------------')
    #print('------------------------------')
    #print(f"total creep strain before: {eqplas}")
    #print('------------------------------')


    if smises > ( 1.0 + self.tolerance ):

      shydro = hydrostaticStress( trial_stress )

      flow = trial_stress

      flow[:3] = flow[:3]-shydro*ones(3)
      flow *= 1.0/smises
      T = self.Tmp
      fl = self.Flux
      #print(type(old_creep))
      #print(smises, T, old_creep, ol_rhoc, ol_rhow, fl)
      #if smises > 299.985430264:
        #smises = 299.985430264

      #Preparing input to model
      input = np.array([smises, T, eqplas, rhoc_old, rhow_old, fl], dtype=float)
      input = input.reshape(-1)
      # print(input)

      #Initial model evaluation
      output = MoEModel.Phi.eval(input)
      # print(output)
      for i in range(3):
        if math.isinf(output[i]) or math.isnan(output[i]):
          file = open("Anomalies.txt", "a")
          str_input = repr(input)
          file.write("MoE Input = " + str_input + "Inf or NaN \n")
          str_output = repr(output)
          file.write("MoE Output = " + str_output + "\n")
      
      if output[1]>1e13 or output[2]>1e13:
        file = open("Anomalies.txt", "a")
        str_input = repr(input)
        file.write("MoE Input = " + str_input + "Unphysical Disc or strain rate or dPhi=0 \n")
        str_output = repr(output)
        file.write("MoE Output = " + str_output + "\n")
      

      deqpl = 0
      dt = self.solverStat.dtime
      rhs  = 0.0

      k = 0

      while (abs(rhs) > self.tolerance or k==0):

        k = k+1

        if k > 100:
          break


        rhs   = output[0] * dt * 1e3 - deqpl
        J = MoEModel.Phi.eval_jacobian(input)
        jac = J[0,0] * self.eg3 * dt * 1e3 -1
        deqpl = deqpl - rhs/(jac) #Problem
        #smises = smises-self.eg3*deqpl #change here
        input = np.array([smises-self.eg3*deqpl, self.Tmp, eqplas, rhoc_old, rhow_old, self.Flux]) #change here
        input = input.reshape(-1)
        output = MoEModel.Phi.eval(input)
        for i in range(3):
          if math.isinf(output[i]) or math.isnan(output[i]):
            file = open("Anomalies.txt", "a")
            str_input = repr(input)
            file.write("MoE Input = " + str_input + "Inf or NaN \n")
            str_output = repr(output)
            file.write("MoE Output = " + str_output + "\n")
        
        if output[1]>1e13 or output[2]>1e13 or jac == -1 :
          file = open("Anomalies.txt", "a")
          str_input = repr(input)
          file.write("MoE Input = " + str_input + "Unphysical Disc or strain rate or dPhi=0 \n")
          str_output = repr(output)
          file.write("MoE Output = " + str_output + "\n")
      
        
        # print('------------------------------')
        # print(f"input in the loop: {input}")
        # print(f"output in the loop: {output}")
        # print("small loop - k = ", k, rhs, deqpl, jac, J[0,0])
        # print('------------------------------')
          
      


      eplas[:3] +=  1.5 * flow[:3] * deqpl
      eelas[:3] += -1.5 * flow[:3] * deqpl

      eplas[3:] +=  3.0 * flow[:3] * deqpl
      eelas[3:] += -3.0 * flow[:3] * deqpl

      deplas = zeros(6)
      deplas[:3] +=  1.5 * flow[:3] * deqpl
      deplas[3:] +=  3.0 * flow[:3] * deqpl

      deelas = kinematics.dstrain - deplas

      sigma = sigma_old +  dot(self.ctang, deelas)

      eqplas += deqpl

      rhoc = rhoc_old + output[1] * dt * 1e3
      rhow = rhow_old + output[2] * dt * 1e3


      # ol_tot_deqpl += deqpl

      # print('------------------------------')
      # print(f"creep strain increment: {deqpl}")
      # print(f"total creep strain: {eqplas}")
      # print("small loop last - k = ", k, rhs, deqpl, jac, J[0,0])
      # print("Big loop")
      # print('------------------------------')
      # print("effective_stress = ", smises-self.eg3*deqpl)
      # print("rhoc = ", rhoc)
      # print("rhow = ", rhow)

      # old_creep = ol_tot_deqpl



      # effg   = vonMisesStress( copy.copy(sigma) ) / copy.copy(smises)
      # effg2  = 2.0*effg
      # effg3  = 3.0*effg
      # efflam = 1.0/3.0 * ( self.ebulk3-effg2 )
      # effhdr = self.eg3 - effg3

      # tang[:3,:3] = efflam

      # for i in range(3):
      #   tang[i,i]     += effg2
      #   tang[i+3,i+3] += effg

      # tang += effhdr*outer(flow,flow)

    tang = self.ctang
    #rohc_new = ol_rhoc + output[1] * dt
    #rohw_new = ol_rhow + output[2] * dt

    self.setHistoryParameter( 'eelas' , eelas  )
    self.setHistoryParameter( 'eplas' , eplas  )
    self.setHistoryParameter( 'sigma' , sigma  )
    self.setHistoryParameter( 'eqplas', eqplas )
    # self.setHistoryParameter( 'old_deqpl' , old_creep)
    self.setHistoryParameter( 'rohc' , rhoc )
    self.setHistoryParameter( 'rohw' , rhow )
    # self.setHistoryParameter( 'total_deqpl' , ol_tot_deqpl)


    # Store output eplas

    self.outData[:6] = sigma
    self.outData[6]  = eqplas
    self.outData[7]  = rhoc
    self.outData[8]  = rhow

    return sigma , tang

