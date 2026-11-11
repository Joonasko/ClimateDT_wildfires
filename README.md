# ClimateDT_wildfires
Wildfires application based on the merge of the FWI and WISE applications

DT CLIMATE - Wildfire Use Case

Latest release: v1.0

This repository contains the scripts related to the Wildfires use case of the Climate Adaptation Digital Twin (Climate DT). All the work is being developed in the frame of the Destination Earth initiative from the European Commission, where ECMWF is one of the Entrusted Entities.

LICENSE NOTE: the European Union, represented by the European Commission is the direct and sole owner of the intellectual property rights of these Results.
Description

This repository calculates the Canadian Fire Weather Index (FWI), a widely used index for evaluating and assessing fire weather conditions. The code is adapted from the original algorithm provided by Van Wagner and Pickett in 1985: Van Wagner, C.E.; Pickett, T.L. (1985). Equations and FORTRAN program for the Canadian Forest Fire Weather Index System. Canadian Forestry Service, Petawawa National Forestry Institute, Chalk River, Ontario. Forestry Technical Report 33, 18 p. Several improvements have been made to adapt the code for global use by reducing regional dependencies. The application also  models the spread of fire at selected test sites, using the WISE (Wildfire Intelligence and Simulation Engine) (Tymstra, C., Bryce, R. W., Wotton, B. M., Taylor, S. W., & Armitage, O. B. (2010). Development and structure of Prometheus: the Canadian wildland fire growth simulation model. Natural Resources Canada, Canadian Forest Service, Northern Forestry Centre, Information Report NOR-X-417.(Edmonton, AB), 9.). WISE is a fire spread model (https://github.com/WISE-Developers) based on the Canadian FWI and FBP (Fire Behaviour Prediction) (Hirsch, K. G. (1996). Canadian forest fire behavior prediction (FBP) system: user's guide. Special Report 7.) indexes.

