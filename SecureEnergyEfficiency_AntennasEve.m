% close all;
% clear all;
% clc;

R=3;
RazaoR=0.01:0.05:1;

xFonte=100;
xDestino=200;
xRelay=150;
xEve=275;

dAR=sqrt((xFonte-xRelay).^2);
dAB=sqrt((xFonte-xDestino).^2);
dRB=sqrt((xRelay-xDestino).^2);
dAE=sqrt((xFonte-xEve).^2);
dRE=sqrt((xRelay-xEve).^2);

alpha = 3;           % Path Loss

N0dBm = -174;
N0 = 10^((N0dBm-30)/10);

B = 10e3;           % Bandwidth
N = N0*B;           % Noise PSD
Rb = R.*B;           % Data rate

b = 1;              % Correcting factor
Vdd = 3;            % Power source
Io = 10e-6;         % current source
n1 = 10;
n2 = 10;
Cp = 1e-12;         % Parasitic Capacitance
fcor = 1e6;         % Corner frequency
Lmin = 0.5e-6;      % Min chan. lenght for current CMOS technology

% TX Circuit
PDAC = b*(0.5*Vdd*Io*(2^n1 - 1) + n1*Cp*(2*B + fcor)*Vdd^2);    % DAC
Pmix = 30.0e-3;     % Mixer
Pfilt = 2.5e-3;     % TX filters

% Sync Circuit
Psyn = 50e-3;       % Freq. Synthetizer

% RX Circuit
PLNA = 20e-3;       % Low-Noise Amplifier
PIFA = 3e-3;        % Intermediate Freq. Amplifier
Pfilr = 2.5e-3;     % RX filters
PADC = (3*Vdd^2*Lmin*(2*B + fcor))/(10^(-0.1525*n2 + 4.838));   % ADC


% Potencia Consumida
P_TX = @(x) x*(PDAC + Pmix + Pfilt + Psyn);
P_RX = @(x) x*(PLNA + Pmix + PIFA + Pfilr + PADC + Psyn);

Ps_dB = -5:1:10;
Ps = 10.^(Ps_dB./10);

Pr_dB = -5:1:10;
Pr = 10.^(Pr_dB./10);

amostras = 1e4;

MldB = 40;              % Link margin
Ml = 10^(MldB/10);

NfdB = 10;              % Noise figure
Nf = 10^(NfdB/10);

GdBi = 5;               % Antenna gains: G = Gt*Gr
G = 10^(GdBi/10);

fc = 2.5e9;             % Carrier frequency
lambda = 3e8/fc;

M = 2^R;
csi = 3*((sqrt(M) - 1)/(sqrt(M) + 1));  % peak-to-average ratio (M-QAM only)
eta = 0.35;
eff = csi/eta - 1;     % Power Amplifier efficiency

outageAlvo=1e-1;

kAB = G*lambda^2/((4*pi)^2*dAB.^alpha*Ml*Nf);
kAR = G*lambda^2/((4*pi)^2*dAR.^alpha*Ml*Nf);
kRB = G*lambda^2/((4*pi)^2*dRB.^alpha*Ml*Nf);
kRE = G*lambda^2/((4*pi)^2*dRE.^alpha*Ml*Nf);
kAE = G*lambda^2/((4*pi)^2*dAE.^alpha*Ml*Nf);

%CSI-DF

hAB_22 = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));
          abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));];
      
hAR = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))];

hRB = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));
       abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))
       abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))];

hAE = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))];

hRE = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))];

hAB_TAS = max([hAB_22(1,1,:).^2 + hAB_22(1,2,:).^2 hAB_22(2,1,:).^2 + hAB_22(2,2,:).^2]);

hRB_TAS = max([hRB(1,1,:).^2 + hRB(1,2,:).^2 hRB(2,1,:).^2 + hRB(2,2,:).^2 hRB(3,1,:).^2 + hRB(3,2,:).^2]); 

%Caso CJ:
hAB_CJ = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));
             abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));];

hAB_TAS_CJ = max([hAB_CJ(1,1,:).^2+hAB_CJ(1,2,:).^2 hAB_CJ(2,1,:).^2+hAB_CJ(2,2,:).^2]);

% Alocação de Taxa e Potência

EficienciaAloc_CSIDF_R_22=[];
EficienciaAloc_CSIDF_R_32=[];
EficienciaAloc_CSIDF_R_42=[];
EficienciaAloc_CSIDF_R_52=[];
EficienciaAloc_CSIDF_R_62=[];
EficienciaAloc_CSIDF_R_72=[];
EficienciaAloc_CSIDF_R_82=[];

EficienciaAloc_CJ_R_21=[];
EficienciaAloc_CJ_R_31=[];
EficienciaAloc_CJ_R_41=[];
EficienciaAloc_CJ_R_51=[];
EficienciaAloc_CJ_R_61=[];
EficienciaAloc_CJ_R_71=[];
EficienciaAloc_CJ_R_81=[];

for a=1:length(RazaoR)
    RazaoR(a)
    
    EficienciaAloc_CSIDF_Potencia_22=[];
    EficienciaAloc_CSIDF_Potencia_32=[];
    EficienciaAloc_CSIDF_Potencia_42=[];
    EficienciaAloc_CSIDF_Potencia_52=[];
    EficienciaAloc_CSIDF_Potencia_62=[];
    EficienciaAloc_CSIDF_Potencia_72=[];
    EficienciaAloc_CSIDF_Potencia_82=[];
    
    EficienciaAloc_CJ_Potencia_21=[];
    EficienciaAloc_CJ_Potencia_31=[];
    EficienciaAloc_CJ_Potencia_41=[];
    EficienciaAloc_CJ_Potencia_51=[];
    EficienciaAloc_CJ_Potencia_61=[];
    EficienciaAloc_CJ_Potencia_71=[];
    EficienciaAloc_CJ_Potencia_81=[];
    
    for i=1:length(Ps) 
%         Ps(i)
        for j=1:length(Pr)
            
%             Pr(j)
            gAB=Ps(i)*kAB/N;
            gAR=Ps(i)*kAR/N;
            gRB=Pr(j)*kRB/N;
            gAE=Ps(i)*kAE/N;
            gRE=Pr(j)*kRE/N;

            
            %CSI-DF
            
            PCooperativo=gAR/(gAB + gAR);
            Pt=(1+eff)*Ps(i)+P_TX(2*1)+P_RX(2*2)+(1-PCooperativo)*((1+eff)*Ps(i))+(PCooperativo)*((1+eff)*Pr(j)+P_RX(1*3));
            
            gABi=gAB*hAB_TAS;
            
            gARi=gAR.*(hAR(1,1,:).^2+hAR(1,2,:).^2+hAR(1,3,:).^2);
            gRBi=gRB*hRB_TAS;
            gAEi=gAE*(hAE(1,1,:).^2 + hAE(1,2,:).^2);
            gREi=gRE*(hRE(1,1,:).^2 + hRE(1,2,:).^2); 
            
            Cb_SDF=log2(1+2*gABi);
            Ce_SDF=log2(1+gREi+gAEi);
            outDireto_22=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gARi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoAR=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gABi+gRBi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoB_22=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            OutageMC_22=outDireto_22.*(outCooperativoAR+(1-outCooperativoAR).*(outCooperativoB_22)); 
 
            gAEi=gAE*(hAE(1,1,:).^2 + hAE(1,2,:).^2 + hAE(1,3,:).^2);
            gREi=gRE*(hRE(1,1,:).^2 + hRE(1,2,:).^2 + hRE(1,3,:).^2); 
            
            Cb_SDF=log2(1+2*gABi);
            Ce_SDF=log2(1+gREi+gAEi);
            outDireto_32=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gARi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoAR=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gABi+gRBi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoB_32=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            OutageMC_32=outDireto_32.*(outCooperativoAR+(1-outCooperativoAR).*(outCooperativoB_32)); 
            
            gAEi=gAE*(hAE(1,1,:).^2 + hAE(1,2,:).^2 + hAE(1,3,:).^2 + hAE(1,4,:).^2);
            gREi=gRE*(hRE(1,1,:).^2 + hRE(1,2,:).^2 + hRE(1,3,:).^2 + hRE(1,4,:).^2); 
            
            Cb_SDF=log2(1+2*gABi);
            Ce_SDF=log2(1+gREi+gAEi);
            outDireto_42=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gARi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoAR=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gABi+gRBi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoB_42=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            OutageMC_42=outDireto_42.*(outCooperativoAR+(1-outCooperativoAR).*(outCooperativoB_42)); 
            
            gAEi=gAE*(hAE(1,1,:).^2 + hAE(1,2,:).^2 + hAE(1,3,:).^2 + hAE(1,4,:).^2 + hAE(1,5,:).^2);
            gREi=gRE*(hRE(1,1,:).^2 + hRE(1,2,:).^2 + hRE(1,3,:).^2 + hRE(1,4,:).^2 + hRE(1,5,:).^2); 
            
            Cb_SDF=log2(1+2*gABi);
            Ce_SDF=log2(1+gREi+gAEi);
            outDireto_52=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gARi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoAR=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gABi+gRBi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoB_52=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            OutageMC_52=outDireto_52.*(outCooperativoAR+(1-outCooperativoAR).*(outCooperativoB_52)); 

            gAEi=gAE*(hAE(1,1,:).^2 + hAE(1,2,:).^2 + hAE(1,3,:).^2 + hAE(1,4,:).^2 + hAE(1,5,:).^2 + hAE(1,6,:).^2);
            gREi=gRE*(hRE(1,1,:).^2 + hRE(1,2,:).^2 + hRE(1,3,:).^2 + hRE(1,4,:).^2 + hRE(1,5,:).^2 + hRE(1,6,:).^2); 
            
            Cb_SDF=log2(1+2*gABi);
            Ce_SDF=log2(1+gREi+gAEi);
            outDireto_62=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gARi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoAR=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gABi+gRBi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoB_62=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            OutageMC_62=outDireto_62.*(outCooperativoAR+(1-outCooperativoAR).*(outCooperativoB_62)); 
            
            gAEi=gAE*(hAE(1,1,:).^2 + hAE(1,2,:).^2 + hAE(1,3,:).^2 + hAE(1,4,:).^2 + hAE(1,5,:).^2 + hAE(1,6,:).^2 + hAE(1,7,:).^2);
            gREi=gRE*(hRE(1,1,:).^2 + hRE(1,2,:).^2 + hRE(1,3,:).^2 + hRE(1,4,:).^2 + hRE(1,5,:).^2 + hRE(1,6,:).^2 + hRE(1,7,:).^2); 
            
            Cb_SDF=log2(1+2*gABi);
            Ce_SDF=log2(1+gREi+gAEi);
            outDireto_72=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gARi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoAR=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gABi+gRBi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoB_72=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras; 
            
            OutageMC_72=outDireto_72.*(outCooperativoAR+(1-outCooperativoAR).*(outCooperativoB_72)); 
            
            gAEi=gAE*(hAE(1,1,:).^2 + hAE(1,2,:).^2 + hAE(1,3,:).^2 + hAE(1,4,:).^2 + hAE(1,5,:).^2 + hAE(1,6,:).^2 + hAE(1,7,:).^2 + hAE(1,8,:).^2);
            gREi=gRE*(hRE(1,1,:).^2 + hRE(1,2,:).^2 + hRE(1,3,:).^2 + hRE(1,4,:).^2 + hRE(1,5,:).^2 + hRE(1,6,:).^2 + hRE(1,7,:).^2 + hRE(1,8,:).^2);
            
            Cb_SDF=log2(1+2*gABi);
            Ce_SDF=log2(1+gREi+gAEi);
            outDireto_82=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gARi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoAR=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            Cb_SDF=log2(1+gABi+gRBi);
            Ce_SDF=log2(1+gREi+gAEi);
            outCooperativoB_82=sum(Cb_SDF-Ce_SDF<2*RazaoR(a)*R)/amostras;
            
            OutageMC_82=outDireto_82.*(outCooperativoAR+(1-outCooperativoAR).*(outCooperativoB_82));
 
            if(OutageMC_22<outageAlvo)
                EficienciaCalculada=((R*RazaoR(a)*(1-OutageMC_22))/Pt);
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CSIDF_Potencia_22 = [EficienciaAloc_CSIDF_Potencia_22; EficienciaCalculada];
            
            if(OutageMC_32<outageAlvo)
                EficienciaCalculada=((R*RazaoR(a)*(1-OutageMC_32))/Pt);
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CSIDF_Potencia_32 = [EficienciaAloc_CSIDF_Potencia_32; EficienciaCalculada];
            
            if(OutageMC_42<outageAlvo)
                EficienciaCalculada=((R*RazaoR(a)*(1-OutageMC_42))/Pt);
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CSIDF_Potencia_42 = [EficienciaAloc_CSIDF_Potencia_42; EficienciaCalculada];
            
            if(OutageMC_52<outageAlvo)
                EficienciaCalculada=((R*RazaoR(a)*(1-OutageMC_52))/Pt);
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CSIDF_Potencia_52 = [EficienciaAloc_CSIDF_Potencia_52; EficienciaCalculada];
            
            if(OutageMC_62<outageAlvo)
                EficienciaCalculada=((R*RazaoR(a)*(1-OutageMC_62))/Pt);
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CSIDF_Potencia_62 = [EficienciaAloc_CSIDF_Potencia_62; EficienciaCalculada];
            
            if(OutageMC_72<outageAlvo)
                EficienciaCalculada=((R*RazaoR(a)*(1-OutageMC_72))/Pt);
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CSIDF_Potencia_72 = [EficienciaAloc_CSIDF_Potencia_72; EficienciaCalculada];

            if(OutageMC_82<outageAlvo)
                EficienciaCalculada=((R*RazaoR(a)*(1-OutageMC_82))/Pt);
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CSIDF_Potencia_82 = [EficienciaAloc_CSIDF_Potencia_82; EficienciaCalculada];
            
            %AN
            gammaI=gRE/3;
            gI=gammaI*(gamrnd(1,1,1,1,amostras)+gamrnd(1,1,1,1,amostras)+gamrnd(1,1,1,1,amostras));
            
            Pt=(1+eff)*2*Ps(i)+P_TX(2*1+2*3)+P_RX(2*2)+(1+eff)*2*Pr(j);
            
            hAE_CJ_MRC=(hAE(1,1,:).^2 + hAE(1,2,:).^2);
            
            Cb = log2(1 + 2*gAB*hAB_TAS_CJ);            
            Ce=log2(1 + 2*((gAE.*hAE_CJ_MRC)./gI));
            Outage_CJ_MC_21=sum(Cb-Ce<2*R*RazaoR(a))/amostras;
            
            hAE_CJ_MRC=(hAE(1,1,:).^2 + hAE(1,2,:).^2 + hAE(1,3,:).^2);
            
            Cb = log2(1 + 2*gAB*hAB_TAS_CJ);            
            Ce=log2(1 + 2*((gAE.*hAE_CJ_MRC)./gI));
            Outage_CJ_MC_31=sum(Cb-Ce<2*R*RazaoR(a))/amostras;
            
            hAE_CJ_MRC=(hAE(1,1,:).^2 + hAE(1,2,:).^2 + hAE(1,3,:).^2 + hAE(1,4,:).^2);
            
            Cb = log2(1 + 2*gAB*hAB_TAS_CJ);            
            Ce=log2(1 + 2*((gAE.*hAE_CJ_MRC)./gI));
            Outage_CJ_MC_41=sum(Cb-Ce<2*R*RazaoR(a))/amostras;
            
            hAE_CJ_MRC=(hAE(1,1,:).^2 + hAE(1,2,:).^2 + hAE(1,3,:).^2 + hAE(1,4,:).^2 + hAE(1,5,:).^2);
            
            Cb = log2(1 + 2*gAB*hAB_TAS_CJ);            
            Ce=log2(1 + 2*((gAE.*hAE_CJ_MRC)./gI));
            Outage_CJ_MC_51=sum(Cb-Ce<2*R*RazaoR(a))/amostras;
            
            hAE_CJ_MRC=(hAE(1,1,:).^2 + hAE(1,2,:).^2 + hAE(1,3,:).^2 + hAE(1,4,:).^2 + hAE(1,5,:).^2 + hAE(1,6,:).^2);
            
            Cb = log2(1 + 2*gAB*hAB_TAS_CJ);            
            Ce=log2(1 + 2*((gAE.*hAE_CJ_MRC)./gI));
            Outage_CJ_MC_61=sum(Cb-Ce<2*R*RazaoR(a))/amostras;
            
            hAE_CJ_MRC=(hAE(1,1,:).^2 + hAE(1,2,:).^2 + hAE(1,3,:).^2 + hAE(1,4,:).^2 + hAE(1,5,:).^2 + hAE(1,6,:).^2 + hAE(1,7,:).^2);
            
            Cb = log2(1 + 2*gAB*hAB_TAS_CJ);            
            Ce=log2(1 + 2*((gAE.*hAE_CJ_MRC)./gI));
            Outage_CJ_MC_71=sum(Cb-Ce<2*R*RazaoR(a))/amostras;
            
            hAE_CJ_MRC=(hAE(1,1,:).^2 + hAE(1,2,:).^2 + hAE(1,3,:).^2 + hAE(1,4,:).^2 + hAE(1,5,:).^2 + hAE(1,6,:).^2 + hAE(1,7,:).^2 + hAE(1,8,:).^2);

            Cb = log2(1 + 2*gAB*hAB_TAS_CJ);            
            Ce=log2(1 + 2*((gAE.*hAE_CJ_MRC)./gI));
            Outage_CJ_MC_81=sum(Cb-Ce<2*R*RazaoR(a))/amostras;
            
            if(Outage_CJ_MC_21<outageAlvo)
                EficienciaCalculada=(R*RazaoR(a)*(1-Outage_CJ_MC_21))/Pt;
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CJ_Potencia_21 = [EficienciaAloc_CJ_Potencia_21; EficienciaCalculada];
            
            if(Outage_CJ_MC_31<outageAlvo)
                EficienciaCalculada=(R*RazaoR(a)*(1-Outage_CJ_MC_31))/Pt;
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CJ_Potencia_31 = [EficienciaAloc_CJ_Potencia_31; EficienciaCalculada];
            
            if(Outage_CJ_MC_41<outageAlvo)
                EficienciaCalculada=(R*RazaoR(a)*(1-Outage_CJ_MC_41))/Pt;
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CJ_Potencia_41 = [EficienciaAloc_CJ_Potencia_41; EficienciaCalculada];
            
            if(Outage_CJ_MC_51<outageAlvo)
                EficienciaCalculada=(R*RazaoR(a)*(1-Outage_CJ_MC_51))/Pt;
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CJ_Potencia_51 = [EficienciaAloc_CJ_Potencia_51; EficienciaCalculada];
            
            if(Outage_CJ_MC_61<outageAlvo)
                EficienciaCalculada=(R*RazaoR(a)*(1-Outage_CJ_MC_61))/Pt;
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CJ_Potencia_61 = [EficienciaAloc_CJ_Potencia_61; EficienciaCalculada];
            
            if(Outage_CJ_MC_71<outageAlvo)
                EficienciaCalculada=(R*RazaoR(a)*(1-Outage_CJ_MC_71))/Pt;
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CJ_Potencia_71 = [EficienciaAloc_CJ_Potencia_71; EficienciaCalculada];  

	    if(Outage_CJ_MC_81<outageAlvo)
                EficienciaCalculada=(R*RazaoR(a)*(1-Outage_CJ_MC_81))/Pt;
            else
                EficienciaCalculada=0;
            end

            EficienciaAloc_CJ_Potencia_81 = [EficienciaAloc_CJ_Potencia_81; EficienciaCalculada];        
        end
    end
    [val index]=max(EficienciaAloc_CSIDF_Potencia_22,[],1);
    EficienciaAloc_CSIDF_R_22(a)=mean(val);
    
    [val index]=max(EficienciaAloc_CSIDF_Potencia_32,[],1);
    EficienciaAloc_CSIDF_R_32(a)=mean(val);
    
    [val index]=max(EficienciaAloc_CSIDF_Potencia_42,[],1);
    EficienciaAloc_CSIDF_R_42(a)=mean(val);
    
    [val index]=max(EficienciaAloc_CSIDF_Potencia_52,[],1);
    EficienciaAloc_CSIDF_R_52(a)=mean(val);
   
    [val index]=max(EficienciaAloc_CSIDF_Potencia_62,[],1);
    EficienciaAloc_CSIDF_R_62(a)=mean(val);
    
    [val index]=max(EficienciaAloc_CSIDF_Potencia_72,[],1);
    EficienciaAloc_CSIDF_R_72(a)=mean(val);

    [val index]=max(EficienciaAloc_CSIDF_Potencia_82,[],1);
    EficienciaAloc_CSIDF_R_82(a)=mean(val);
    
    [val index]=max(EficienciaAloc_CJ_Potencia_21,[],1);
    EficienciaAloc_CJ_R_21(a)=mean(val);
    
    [val index]=max(EficienciaAloc_CJ_Potencia_31,[],1);
    EficienciaAloc_CJ_R_31(a)=mean(val);
    
    [val index]=max(EficienciaAloc_CJ_Potencia_41,[],1);
    EficienciaAloc_CJ_R_41(a)=mean(val);
    
    [val index]=max(EficienciaAloc_CJ_Potencia_51,[],1);
    EficienciaAloc_CJ_R_51(a)=mean(val);
   
    [val index]=max(EficienciaAloc_CJ_Potencia_61,[],1);
    EficienciaAloc_CJ_R_61(a)=mean(val);
    
    [val index]=max(EficienciaAloc_CJ_Potencia_71,[],1);
    EficienciaAloc_CJ_R_71(a)=mean(val);

    [val index]=max(EficienciaAloc_CJ_Potencia_81,[],1);
    EficienciaAloc_CJ_R_81(a)=mean(val);
end

Eficiencia_CSI_DF_22_nE=max(EficienciaAloc_CSIDF_R_22);
Eficiencia_CSI_DF_32_nE=max(EficienciaAloc_CSIDF_R_32);
Eficiencia_CSI_DF_42_nE=max(EficienciaAloc_CSIDF_R_42);
Eficiencia_CSI_DF_52_nE=max(EficienciaAloc_CSIDF_R_52);
Eficiencia_CSI_DF_62_nE=max(EficienciaAloc_CSIDF_R_62);
Eficiencia_CSI_DF_72_nE=max(EficienciaAloc_CSIDF_R_72);
Eficiencia_CSI_DF_82_nE=max(EficienciaAloc_CSIDF_R_82);

Eficiencia_CJ_21_nE=max(EficienciaAloc_CJ_R_21);
Eficiencia_CJ_31_nE=max(EficienciaAloc_CJ_R_31);
Eficiencia_CJ_41_nE=max(EficienciaAloc_CJ_R_41);
Eficiencia_CJ_51_nE=max(EficienciaAloc_CJ_R_51);
Eficiencia_CJ_61_nE=max(EficienciaAloc_CJ_R_61);
Eficiencia_CJ_71_nE=max(EficienciaAloc_CJ_R_71);
Eficiencia_CJ_81_nE=max(EficienciaAloc_CJ_R_81);

%%
nR=3:9;

Matriz_CSIDF_nR=[Eficiencia_CSI_DF_22 Eficiencia_CSI_DF_32 Eficiencia_CSI_DF_42 Eficiencia_CSI_DF_52 Eficiencia_CSI_DF_62 Eficiencia_CSI_DF_72 Eficiencia_CSI_DF_82];
Matriz_CJ_nR=[Eficiencia_CJ_21 Eficiencia_CJ_31 Eficiencia_CJ_41 Eficiencia_CJ_51 Eficiencia_CJ_61 Eficiencia_CJ_71 Eficiencia_CJ_81];

y1 = hist(Matriz_CSIDF_nR, nR);   
y2 = hist(Matriz_CJ_nR, nR);

figure(1);

subplot(2,1,1)

b=bar(nR,[Matriz_CSIDF_nR; Matriz_CJ_nR]', 'group');
b(1).FaceColor = 'blue';
b(2).FaceColor = 'red';
hold on;
xlabel('Antennas at the relay','fontsize',13);
ylabel('$\eta_{s} \rm\bf(bits\,seguros/J/Hz)$','Interpreter','LaTeX','Fontsize',14);
legend('CSI-DF', 'AN', 'Location', 'northwest');
ylim([0 0.41])
title('Fig. 17(a)')
grid on;

nE=2:8;

Matriz_CSIDF_nE=[Eficiencia_CSI_DF_22_nE Eficiencia_CSI_DF_32_nE Eficiencia_CSI_DF_42_nE Eficiencia_CSI_DF_52_nE Eficiencia_CSI_DF_62_nE Eficiencia_CSI_DF_72_nE Eficiencia_CSI_DF_82_nE];
Matriz_CJ_nE=[Eficiencia_CJ_21_nE Eficiencia_CJ_31_nE Eficiencia_CJ_41_nE Eficiencia_CJ_51_nE Eficiencia_CJ_61_nE Eficiencia_CJ_71_nE Eficiencia_CJ_81_nE];

y1 = hist(Matriz_CSIDF_nE, nE);   
y2 = hist(Matriz_CJ_nE, nE);

subplot(2,1,2)

b=bar(nE,[Matriz_CSIDF_nE; Matriz_CJ_nE]', 'group');
b(1).FaceColor = 'blue';
b(2).FaceColor = 'red';
hold on;
xlabel('Antennas at Eve','fontsize',13);
title('Fig. 17(b)')
% ylabel('$\eta_{s} \rm\bf(bits\,seguros/J/Hz)$','Interpreter','LaTeX','Fontsize',14);
legend('CSI-DF', 'AN', 'Location', 'northwest');
grid on;
% ylim([0 2.01])
