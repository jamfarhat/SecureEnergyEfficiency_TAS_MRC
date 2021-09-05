close all;
clear all;
clc;
 
R=3;
alpha = 3; % Path Loss

Ps_dB = -5:1:10;
Ps = 10.^(Ps_dB./10);

Pr_dB = -5:1:10;
Pr = 10.^(Pr_dB./10);    
 
N0dBm = -174;
N0 = 10^((N0dBm-30)/10);
 
B = 10e3;           % Bandwidth
N = N0*B;           % Noise PSD
Rb = R*B;           % Data rate
 
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
 
M = 2^R;
csi = 3*((sqrt(M) - 1)/(sqrt(M) + 1));  % peak-to-average ratio (M-QAM only)
eta = 0.35;
eff = csi/eta - 1;     % Power Amplifier efficiency
 
MldB = 40;              % Link margin
Ml = 10^(MldB/10);
 
NfdB = 10;              % Noise figure
Nf = 10^(NfdB/10);
 
GdBi = 5;               % Antenna gains: G = Gt*Gr
G = 10^(GdBi/10);
 
fc = 2.5e9;             % Carrier frequency
lambda = 3e8/fc;
 
xFonte=100;
xDestino=200;
xRelay=150;
xEve=xRelay+1:25:xRelay+301;
 
dsr=sqrt((xFonte-xRelay).^2);
dsd=sqrt((xFonte-xDestino).^2);
drd=sqrt((xRelay-xDestino).^2);
dse=sqrt((xFonte-xEve).^2);
dre=sqrt((xRelay-xEve).^2);
 
ksd = G*lambda^2/((4*pi)^2*dsd^alpha*Ml*Nf);
ksr = G*lambda^2/((4*pi)^2*dsr^alpha*Ml*Nf);
krd = G*lambda^2/((4*pi)^2*drd^alpha*Ml*Nf);
kre = G*lambda^2./((4*pi)^2.*dre.^alpha*Ml*Nf);
kse = G*lambda^2./((4*pi)^2.*dse.^alpha*Ml*Nf);

amostras=1e6;
theta=0.1:0.05:1;

outageAlvo=1e-1;

%Caso 1:
hsd = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));
       abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));];

hsr = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))];

hrd = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));
       abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))];

hse = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))];

hre = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))];

hsd_tas = max([hsd(1,1,:).^2 + hsd(1,2,:).^2 hsd(2,1,:).^2 + hsd(2,2,:).^2]);

hrd_tas = max([hrd(1,1,:).^2 + hrd(1,2,:).^2 hrd(2,1,:).^2 + hrd(2,2,:).^2]);  
% 
for IndiceDistancia=1:length(dre)
    disp(['CSI-RC - d_{RE}=' num2str(dre(IndiceDistancia))])
    
    IndiceTheta=5;
    IndicePs=2;
    IndicePr=2;
   
    gSD=Ps(IndicePs)*ksd/N;
    gSR=Ps(IndicePs)*ksr/N;
    gRD=Pr(IndicePr)*krd/N;
    gSE=Ps(IndicePs)*kse(IndiceDistancia)/N;
    gRE=Pr(IndicePr)*kre(IndiceDistancia)/N;

    %Caso 1:
    gSDi=gSD*hsd_tas;
    gSRi=gSR.*(hsr(1,1,:).^2+hsr(1,2,:).^2);
    gRDi=gRD*hrd_tas;
    gSEi=gSE*(hse(1,1,:).^2 + hse(1,2,:).^2);
    gREi=gRE*(hre(1,1,:).^2 + hre(1,2,:).^2);

    Cb_SDF=log2(1+2*gSDi);
    Ce_SDF=log2(1+2*gSEi);
    outDireto=sum(Cb_SDF-Ce_SDF<2*theta(IndiceTheta)*R)/amostras;

    Cb_SDF=log2(1+gSRi);
    Ce_SDF=log2(1+gREi+gSEi);
    outCooperativoAR=sum(Cb_SDF-Ce_SDF<2*theta(IndiceTheta)*R)/amostras;

    Cb_SDF=log2(1+gSDi+gRDi);
    Ce_SDF=log2(1+gREi+gSEi);
    outCooperativoB=sum(Cb_SDF-Ce_SDF<2*theta(IndiceTheta)*R)/amostras;

    OutageEscolhida=outDireto*(outCooperativoAR+(1-outCooperativoAR)*(outCooperativoB)); 

    PCooperativo=gSR/(gSD + gSR);
    Pt=(1+eff)*Ps(IndicePs)+P_TX(2*1)+P_RX(2*2)+(1-PCooperativo)*(1+eff)*Ps(IndicePs)+(PCooperativo)*((1+eff)*Pr(IndicePr)+P_RX(1*2));

    if(OutageEscolhida<outageAlvo)
        EficienciaCalculada=(R*theta(IndiceTheta)*(1-OutageEscolhida))/Pt;
    else
        EficienciaCalculada=0;
    end 

    EficienciaCSIRC(IndiceDistancia)=EficienciaCalculada;
    
%     Monte Carlo
    
    Rs=theta(IndiceTheta)*R;
    Ns = 2;
    Nr = 2;
    Nd = 2;
    Ne = 2;

    alfa=((2^(-2*Rs))-1);
    beta=(2*(2^(-2*Rs)));
    S0=0;
    for(k=0:Ns-1)
        S1=0;
        for(iu1=0:k)
            Prob1=((factorial(Ne-1))/((gSE^(-Ne))))*(factorial(iu1+Nd-1)*(((k+1)/gSD)^(-(iu1+Nd))));
            
            S2=0;
            for(m=0:Ne-1)
                S3=0;
                for(v=0:m)
                   S3=S3+nchoosek(m, v)*(alfa^(m-v))*(beta^v)*factorial(v+iu1+Nd-1)*((((k+1)/(gSD))+(beta/gSE))^(-(v+iu1+Nd)));
                end
                S2=S2+((factorial(Ne-1))/((factorial(m))*(gSE^(m-Ne))))*S3;
            end    
            Prob2=exp(-(alfa/gSE))*S2;
            
            S2=0;
            for(w=0:Ne-1)
                S3=0;
                for(v=0:w)
                    S4=0;
                    for(z=0:w-v)
                       Prob3=((factorial(v+Ne-1))/(((1/gSE)-(1/gRE))^(v+Ne)))*(factorial(z+iu1+Nd-1)*((((k+1)/gSD)+(beta/gRE))^(-(z+iu1+Nd))));
                       
                       S5=0;
                       for(m=0:v+Ne-1)
                           S6=0;
                           for(y=0:m)
                               S6=S6+nchoosek(m, y)*(alfa^(m-y))*(beta^y)*factorial(y+z+iu1+Nd-1)*((((k+1)/gSD)+(beta/gSE))^(-(y+z+iu1+Nd)));
                           end
                           S5=S5+(factorial(v+Ne-1)/((factorial(m))*(((1/gSE)-(1/gRE))^(v+Ne-m))))*S6; 
                       end
                       Prob4=exp(-alfa*((1/gSE)-(1/gRE)))*S5;
                       
                       S4=S4+nchoosek(w-v, z)*(alfa^(w-v-z))*(beta^z)*(Prob3-Prob4);
                    end
                    S3=S3+nchoosek(w, v)*((-1)^v)*S4;
                end
                S2=S2+(1/factorial(w))*((1/gRE)^w)*exp(-(alfa/gRE))*S3; 
            end
            ProbSoma=S2;
            
            S1=S1+nchoosek(k, iu1)*((1/factorial(1))^(iu1-0))*((1/(gSD))^(iu1))*(Prob1-Prob2-ProbSoma);
        end
        S0=S0+nchoosek(Ns-1, k)*((-1)^k)*S1;
    end
    OutageDireto=1-(Ns/((gamma(Nd))*(gamma(Ne))*(gSD^Nd)*(gSE^Ne)))*S0;
    
    S0=0;
    for(k=0:Ns-1)
        ni=[k zeros(1,Nd)];
        P0=1;
        for(u=1:Nd-1)
            S1=0;
            for(i_u=0:ni(u))
                ni(u+1)=i_u;
                ValorN=0;
                for(LoopPhi=2:Nd)
                    ValorN=ValorN+ni(LoopPhi);
                end
                
                S2=0;
                for(m=0:Nr-1)
                    ni1=[m zeros(1,Nd)];
                    P1=1;
                    for(u1=1:Nd-1)
                        S3=0;
                        for(i_u1=0:ni1(u1))
                            ni1(u1+1)=i_u1;
                            SomaM=0;
                            for(LoopPhi=2:Nd)
                                SomaM=SomaM+ni1(LoopPhi);
                            end
                            
                            S4=0;
                            for(l=0:Nd+ValorN-1)
                                u=Nd+ValorN-l;
                                v=Nd+SomaM+l;
                                Beta=((k+1)/gSD);
                                alfa=(((m+1)/(gRD))-((k+1)/(gSD)));
                                Prob1=(factorial(Ne-1)/(gSE^(-Ne)))*(((alfa^v)*gamma(u+v))/(v*(alfa+Beta)^(u+v)))*hypergeom([1 u+v], v+1, (alfa)/(alfa+Beta));
                                
                                S5=0;
                                for(p=0:Ne-1)
                                    S6=0;
                                    for(s=0:p)
                                        u=s+Nd+ValorN-l;
                                        v=Nd+SomaM+l;
                                        Beta=((k+1)/gSD)+((2^(-2*Rs))/gSE);
                                        alfa=(((m+1)/(gRD))-((k+1)/(gSD)));
                                        S6=S6+nchoosek(p, s)*(2^(-2*Rs*s))*((2^(-2*Rs)-1)^(p-s))*(((alfa^v)*gamma(u+v))/(v*(alfa+Beta)^(u+v)))*hypergeom([1 u+v], v+1, (alfa)/(alfa+Beta));
                                    end
                                    S5=S5+(factorial(Ne-1)/factorial(p))*(1/(gSE^(p-Ne)))*exp(-(2^(-2*Rs)-1)/(gSE))*S6;
                                end
                                Prob2=S5;
                                
                                S5=0;
                                for(w=0:Ne-1)
                                    S6=0;
                                    for(o=0:w)
                                        S7=0;
                                        for(s=0:w-o)
                                            u=s+Nd+ValorN-l;
                                            v=Nd+SomaM+l;
                                            Beta=((k+1)/gSD)+((2^(-2*Rs))/gRE);
                                            alfa=(((m+1)/(gRD))-((k+1)/(gSD)));
                                            S7=S7+nchoosek(w-o, s)*(2^(-2*Rs*s))*((2^(-2*Rs)-1)^(w-o-s))*(((alfa^v)*gamma(u+v))/(v*(alfa+Beta)^(u+v)))*hypergeom([1 u+v], v+1, (alfa)/(alfa+Beta));
                                        end
                                        
                                        S10=0;
                                        for(t=0:o+Ne-1)
                                            S11=0;
                                            for(x=0:w+t-o)
                                                u2=x+Nd+ValorN-l;
                                                v2=Nd+SomaM+l;
                                                Beta2=((k+1)/gSD)+((2^(-2*Rs))/gSE);
                                                alfa2=(((m+1)/(gRD))-((k+1)/(gSD)));
                                                S11=S11+nchoosek(w+t-o, x)*(2^(-2*Rs*x))*((2^(-2*Rs)-1)^(w+t-o-x))*(((alfa2^v2)*gamma(u2+v2))/(v2*(alfa2+Beta2)^(u2+v2)))*hypergeom([1 u2+v2], v2+1, (alfa2)/(alfa2+Beta2));
                                            end
                                            S10=S10+(factorial(o+Ne-1)/factorial(t))*(((1/gSE)-(1/gRE))^(-(o+Ne-t)))*exp(-(2^(-2*Rs)-1)/(gSE))*S11;
                                        end
                                        
                                        S6=S6+nchoosek(w, o)*((-1)^o)*(((factorial(o+Ne-1))/(((1/gSE)-(1/gRE))^(o+Ne)))*(exp(-(2^(-2*Rs)-1)/(gRE))*S7)-S10);
                                    end
                                    S5=S5+(1/factorial(w))*((1/gRE)^w)*S6;
                                end
                                Prob3=S5;
                                
                                S4=S4+nchoosek(Nd+ValorN-1,l)*((-1)^l)*(alfa^(-(Nd+SomaM+l)))*(Prob1-Prob2-Prob3);
                            end
                            
                            S3=S3+nchoosek(ni1(1), i_u1)*((1/factorial(u1))^(i_u1-ni1(2)))*((1/gRD)^i_u1)*S4;
                        end
                        P1=P1*S3;
                    end
                    S2=S2+nchoosek(Nr-1, m)*((-1)^m)*P1;
                end
                S1=S1+nchoosek(ni(1), i_u)*((1/factorial(u))^(i_u-ni(2)))*((1/gSD)^i_u)*S2;
            end   
            P0=P0*S1;
        end
        S0=S0+nchoosek(Ns-1, k)*((-1)^k)*P0;
    end
    OutageCoopB=1-((Nr*Ns)/((gSE^Ne)*gamma(Ne)*(gamma(Nd)^2)*((gSD*gRD)^Nd)))*S0;
    
    Prob1=0;
    Prob2=0;
    Prob3=0;
    Prob4=0;
    ProbTotal=0;
    Prob1=((factorial(Ne-1)*factorial(Nr-1))/((gSE^(-Ne))*(gSR^(-Nr))));
    
    S0=0;
    for(m=0:Ne-1)
        S1=0;
        for(p=0:m)
           S1=S1+nchoosek(m,p)*(2^(-2*Rs*p))*(((2^(-2*Rs))-1)^(m-p))*factorial(Nr+p-1)*(((1/gSR)+((2^(-2*Rs))/(gSE)))^(-(Nr+p)));
        end
        S0=S0+((factorial(Ne-1))/(factorial(m)))*((1/gSE)^(m-Ne))*exp(-((2^(-2*Rs))-1)/gSE)*S1;
    end
    Prob2=S0;
  
    S0=0;
    for(w=0:Ne-1)
        S1=0;
        for(k=0:w)
            S2=0;
            for(p=0:w-k)
               S2=S2+nchoosek(w-k, p)*(2^(-2*Rs*p))*(((2^(-2*Rs))-1)^(w-k-p))*factorial(Nr+p-1)*((1/gSR)+((2^(-2*Rs))/(gRE)))^(-(Nr+p));
            end
            Prob3=((factorial(k+Ne-1))/(((1/gSE)-(1/gRE))^(k+Ne)))*(exp(-((2^(-2*Rs))-1)/(gRE)))*S2;
            
            S2=0;
            for(o=0:k+Ne-1)
                S3=0;
                for(p=0:w-k+o)
                    S3=S3+nchoosek(w-k+o, p)*(2^(-2*Rs*p))*(((2^(-2*Rs))-1)^(w-k+o-p))*factorial(Nr+p-1)*((1/gSR)+((2^(-2*Rs))/(gSE)))^(-(Nr+p));
                end
                S2=S2+((factorial(k+Ne-1))/(factorial(o)*(((1/gSE)-(1/gRE))^(k+Ne-o))))*exp(-((2^(-2*Rs))-1)/(gSE))*S3;
            end
            Prob4=S2;
            
            S1=S1+nchoosek(w,k)*((-1)^k)*(Prob3-Prob4);
        end
        S0=S0+(1/factorial(w))*((1/gRE)^w)*S1; 
    end
    ProbTotal=S0;
    
    OutageCoopAR=1-(1/(gamma(Ne)*gamma(Nr)*((gSE)^(Ne))*((gSR)^(Nr))))*(Prob1-Prob2-ProbTotal);      
 
    OutageMC=OutageDireto*(OutageCoopAR+(1-OutageCoopAR)*(OutageCoopB));
    
    if(OutageMC<outageAlvo)
        EficienciaCalculada=(R*theta(IndiceTheta)*(1-OutageMC))/Pt;
    else
        EficienciaCalculada=0;
    end 

    EficienciaCSIRC_MC(IndiceDistancia)=EficienciaCalculada;
end 

EficienciaCSIRC_AlocacaoPotencia=[];
for IndiceDistancia=1:length(dre)
    disp(['CSI-RC - d_{RE}=' num2str(dre(IndiceDistancia))])
    
    IndiceTheta=5;
    
    EficienciaCSIRC1_Ps=[];
    for(IndicePs=1:length(Ps))
        EficienciaCSIRC1_Pr=[];
        for(IndicePr=1:length(Pr))
            gSD=Ps(IndicePs)*ksd/N;
            gSR=Ps(IndicePs)*ksr/N;
            gRD=Pr(IndicePr)*krd/N;
            gSE=Ps(IndicePs)*kse(IndiceDistancia)/N;
            gRE=Pr(IndicePr)*kre(IndiceDistancia)/N;

        %Caso 1:
            gSDi=gSD*hsd_tas;
            gSRi=gSR.*(hsr(1,1,:).^2+hsr(1,2,:).^2);
            gRDi=gRD*hrd_tas;
            gSEi=gSE*(hse(1,1,:).^2 + hse(1,2,:).^2);
            gREi=gRE*(hre(1,1,:).^2 + hre(1,2,:).^2);

            Cb_SDF=log2(1+2*gSDi);
            Ce_SDF=log2(1+2*gSEi);
            outDireto=sum(Cb_SDF-Ce_SDF<2*theta(IndiceTheta)*R)/amostras;

            Cb_SDF=log2(1+gSRi);
            Ce_SDF=log2(1+gREi+gSEi);
            outCooperativoAR=sum(Cb_SDF-Ce_SDF<2*theta(IndiceTheta)*R)/amostras;

            Cb_SDF=log2(1+gSDi+gRDi);
            Ce_SDF=log2(1+gREi+gSEi);
            outCooperativoB=sum(Cb_SDF-Ce_SDF<2*theta(IndiceTheta)*R)/amostras;

            OutageEscolhida=outDireto*(outCooperativoAR+(1-outCooperativoAR)*(outCooperativoB)); 

            PCooperativo=gSR/(gSD + gSR);
            Pt=(1+eff)*Ps(IndicePs)+P_TX(2*1)+P_RX(2*2)+(1-PCooperativo)*(1+eff)*Ps(IndicePs)+(PCooperativo)*((1+eff)*Pr(IndicePr)+P_RX(1*2));

            if(OutageEscolhida<outageAlvo)
                EficienciaCalculada=(R*theta(IndiceTheta)*(1-OutageEscolhida))/Pt;
            else
                EficienciaCalculada=0;
            end

            EficienciaCSIRC1_Pr = [EficienciaCSIRC1_Pr EficienciaCalculada];  
        end
        [val index]=max(EficienciaCSIRC1_Pr);
        EficienciaCSIRC1_Ps=[EficienciaCSIRC1_Ps val];
        
        ValoresPr(IndicePs)=Pr(index);
    end 
    [val index]=max(EficienciaCSIRC1_Ps);
    EficienciaCSIRC_AlocacaoPotencia=[EficienciaCSIRC_AlocacaoPotencia val];
    
    ValorPr=ValoresPr(index);
    ValorPs=Ps(index);
    
    gSD=ValorPs*ksd/N;
    gSR=ValorPs*ksr/N;
    gRD=ValorPr*krd/N;
    gSE=ValorPs*kse(IndiceDistancia)/N;
    gRE=ValorPr*kre(IndiceDistancia)/N;
    
    PCooperativo=gSR/(gSD + gSR);
    Pt=(1+eff)*ValorPs+P_TX(2*1)+P_RX(2*2)+(1-PCooperativo)*(1+eff)*ValorPs+(PCooperativo)*((1+eff)*ValorPr+P_RX(1*2));
    
    Rs=theta(IndiceTheta)*R;
    Ns = 2;
    Nr = 2;
    Nd = 2;
    Ne = 2;
% 
    alfa=((2^(-2*Rs))-1);
    beta=(2*(2^(-2*Rs)));
    S0=0;
    for(k=0:Ns-1)
        S1=0;
        for(iu1=0:k)
            Prob1=((factorial(Ne-1))/((gSE^(-Ne))))*(factorial(iu1+Nd-1)*(((k+1)/gSD)^(-(iu1+Nd))));
            
            S2=0;
            for(m=0:Ne-1)
                S3=0;
                for(v=0:m)
                   S3=S3+nchoosek(m, v)*(alfa^(m-v))*(beta^v)*factorial(v+iu1+Nd-1)*((((k+1)/(gSD))+(beta/gSE))^(-(v+iu1+Nd)));
                end
                S2=S2+((factorial(Ne-1))/((factorial(m))*(gSE^(m-Ne))))*S3;
            end    
            Prob2=exp(-(alfa/gSE))*S2;
            
            S2=0;
            for(w=0:Ne-1)
                S3=0;
                for(v=0:w)
                    S4=0;
                    for(z=0:w-v)
                       Prob3=((factorial(v+Ne-1))/(((1/gSE)-(1/gRE))^(v+Ne)))*(factorial(z+iu1+Nd-1)*((((k+1)/gSD)+(beta/gRE))^(-(z+iu1+Nd))));
                       
                       S5=0;
                       for(m=0:v+Ne-1)
                           S6=0;
                           for(y=0:m)
                               S6=S6+nchoosek(m, y)*(alfa^(m-y))*(beta^y)*factorial(y+z+iu1+Nd-1)*((((k+1)/gSD)+(beta/gSE))^(-(y+z+iu1+Nd)));
                           end
                           S5=S5+(factorial(v+Ne-1)/((factorial(m))*(((1/gSE)-(1/gRE))^(v+Ne-m))))*S6; 
                       end
                       Prob4=exp(-alfa*((1/gSE)-(1/gRE)))*S5;
                       
                       S4=S4+nchoosek(w-v, z)*(alfa^(w-v-z))*(beta^z)*(Prob3-Prob4);
                    end
                    S3=S3+nchoosek(w, v)*((-1)^v)*S4;
                end
                S2=S2+(1/factorial(w))*((1/gRE)^w)*exp(-(alfa/gRE))*S3; 
            end
            ProbSoma=S2;
            
            S1=S1+nchoosek(k, iu1)*((1/factorial(1))^(iu1-0))*((1/(gSD))^(iu1))*(Prob1-Prob2-ProbSoma);
        end
        S0=S0+nchoosek(Ns-1, k)*((-1)^k)*S1;
    end
    OutageDireto=1-(Ns/((gamma(Nd))*(gamma(Ne))*(gSD^Nd)*(gSE^Ne)))*S0;
    
    S0=0;
    for(k=0:Ns-1)
        ni=[k zeros(1,Nd)];
        P0=1;
        for(u=1:Nd-1)
            S1=0;
            for(i_u=0:ni(u))
                ni(u+1)=i_u;
                ValorN=0;
                for(LoopPhi=2:Nd)
                    ValorN=ValorN+ni(LoopPhi);
                end
                
                S2=0;
                for(m=0:Nr-1)
                    ni1=[m zeros(1,Nd)];
                    P1=1;
                    for(u1=1:Nd-1)
                        S3=0;
                        for(i_u1=0:ni1(u1))
                            ni1(u1+1)=i_u1;
                            SomaM=0;
                            for(LoopPhi=2:Nd)
                                SomaM=SomaM+ni1(LoopPhi);
                            end
                            
                            S4=0;
                            for(l=0:Nd+ValorN-1)
                                u=Nd+ValorN-l;
                                v=Nd+SomaM+l;
                                Beta=((k+1)/gSD);
                                alfa=(((m+1)/(gRD))-((k+1)/(gSD)));
                                Prob1=(factorial(Ne-1)/(gSE^(-Ne)))*(((alfa^v)*gamma(u+v))/(v*(alfa+Beta)^(u+v)))*hypergeom([1 u+v], v+1, (alfa)/(alfa+Beta));
                                
                                S5=0;
                                for(p=0:Ne-1)
                                    S6=0;
                                    for(s=0:p)
                                        u=s+Nd+ValorN-l;
                                        v=Nd+SomaM+l;
                                        Beta=((k+1)/gSD)+((2^(-2*Rs))/gSE);
                                        alfa=(((m+1)/(gRD))-((k+1)/(gSD)));
                                        S6=S6+nchoosek(p, s)*(2^(-2*Rs*s))*((2^(-2*Rs)-1)^(p-s))*(((alfa^v)*gamma(u+v))/(v*(alfa+Beta)^(u+v)))*hypergeom([1 u+v], v+1, (alfa)/(alfa+Beta));
                                    end
                                    S5=S5+(factorial(Ne-1)/factorial(p))*(1/(gSE^(p-Ne)))*exp(-(2^(-2*Rs)-1)/(gSE))*S6;
                                end
                                Prob2=S5;
                                
                                S5=0;
                                for(w=0:Ne-1)
                                    S6=0;
                                    for(o=0:w)
                                        S7=0;
                                        for(s=0:w-o)
                                            u=s+Nd+ValorN-l;
                                            v=Nd+SomaM+l;
                                            Beta=((k+1)/gSD)+((2^(-2*Rs))/gRE);
                                            alfa=(((m+1)/(gRD))-((k+1)/(gSD)));
                                            S7=S7+nchoosek(w-o, s)*(2^(-2*Rs*s))*((2^(-2*Rs)-1)^(w-o-s))*(((alfa^v)*gamma(u+v))/(v*(alfa+Beta)^(u+v)))*hypergeom([1 u+v], v+1, (alfa)/(alfa+Beta));
                                        end
                                        
                                        S10=0;
                                        for(t=0:o+Ne-1)
                                            S11=0;
                                            for(x=0:w+t-o)
                                                u2=x+Nd+ValorN-l;
                                                v2=Nd+SomaM+l;
                                                Beta2=((k+1)/gSD)+((2^(-2*Rs))/gSE);
                                                alfa2=(((m+1)/(gRD))-((k+1)/(gSD)));
                                                S11=S11+nchoosek(w+t-o, x)*(2^(-2*Rs*x))*((2^(-2*Rs)-1)^(w+t-o-x))*(((alfa2^v2)*gamma(u2+v2))/(v2*(alfa2+Beta2)^(u2+v2)))*hypergeom([1 u2+v2], v2+1, (alfa2)/(alfa2+Beta2));
                                            end
                                            S10=S10+(factorial(o+Ne-1)/factorial(t))*(((1/gSE)-(1/gRE))^(-(o+Ne-t)))*exp(-(2^(-2*Rs)-1)/(gSE))*S11;
                                        end
                                        
                                        S6=S6+nchoosek(w, o)*((-1)^o)*(((factorial(o+Ne-1))/(((1/gSE)-(1/gRE))^(o+Ne)))*(exp(-(2^(-2*Rs)-1)/(gRE))*S7)-S10);
                                    end
                                    S5=S5+(1/factorial(w))*((1/gRE)^w)*S6;
                                end
                                Prob3=S5;
                                
                                S4=S4+nchoosek(Nd+ValorN-1,l)*((-1)^l)*(alfa^(-(Nd+SomaM+l)))*(Prob1-Prob2-Prob3);
                            end
                            
                            S3=S3+nchoosek(ni1(1), i_u1)*((1/factorial(u1))^(i_u1-ni1(2)))*((1/gRD)^i_u1)*S4;
                        end
                        P1=P1*S3;
                    end
                    S2=S2+nchoosek(Nr-1, m)*((-1)^m)*P1;
                end
                S1=S1+nchoosek(ni(1), i_u)*((1/factorial(u))^(i_u-ni(2)))*((1/gSD)^i_u)*S2;
            end   
            P0=P0*S1;
        end
        S0=S0+nchoosek(Ns-1, k)*((-1)^k)*P0;
    end
    OutageCoopB=1-((Nr*Ns)/((gSE^Ne)*gamma(Ne)*(gamma(Nd)^2)*((gSD*gRD)^Nd)))*S0;
    
    Prob1=0;
    Prob2=0;
    Prob3=0;
    Prob4=0;
    ProbTotal=0;
    Prob1=((factorial(Ne-1)*factorial(Nr-1))/((gSE^(-Ne))*(gSR^(-Nr))));
    
    S0=0;
    for(m=0:Ne-1)
        S1=0;
        for(p=0:m)
           S1=S1+nchoosek(m,p)*(2^(-2*Rs*p))*(((2^(-2*Rs))-1)^(m-p))*factorial(Nr+p-1)*(((1/gSR)+((2^(-2*Rs))/(gSE)))^(-(Nr+p)));
        end
        S0=S0+((factorial(Ne-1))/(factorial(m)))*((1/gSE)^(m-Ne))*exp(-((2^(-2*Rs))-1)/gSE)*S1;
    end
    Prob2=S0;
  
    S0=0;
    for(w=0:Ne-1)
        S1=0;
        for(k=0:w)
            S2=0;
            for(p=0:w-k)
               S2=S2+nchoosek(w-k, p)*(2^(-2*Rs*p))*(((2^(-2*Rs))-1)^(w-k-p))*factorial(Nr+p-1)*((1/gSR)+((2^(-2*Rs))/(gRE)))^(-(Nr+p));
            end
            Prob3=((factorial(k+Ne-1))/(((1/gSE)-(1/gRE))^(k+Ne)))*(exp(-((2^(-2*Rs))-1)/(gRE)))*S2;
            
            S2=0;
            for(o=0:k+Ne-1)
                S3=0;
                for(p=0:w-k+o)
                    S3=S3+nchoosek(w-k+o, p)*(2^(-2*Rs*p))*(((2^(-2*Rs))-1)^(w-k+o-p))*factorial(Nr+p-1)*((1/gSR)+((2^(-2*Rs))/(gSE)))^(-(Nr+p));
                end
                S2=S2+((factorial(k+Ne-1))/(factorial(o)*(((1/gSE)-(1/gRE))^(k+Ne-o))))*exp(-((2^(-2*Rs))-1)/(gSE))*S3;
            end
            Prob4=S2;
            
            S1=S1+nchoosek(w,k)*((-1)^k)*(Prob3-Prob4);
        end
        S0=S0+(1/factorial(w))*((1/gRE)^w)*S1; 
    end
    ProbTotal=S0;
    
    OutageCoopAR=1-(1/(gamma(Ne)*gamma(Nr)*((gSE)^(Ne))*((gSR)^(Nr))))*(Prob1-Prob2-ProbTotal);      
 
    OutageMC=OutageDireto*(OutageCoopAR+(1-OutageCoopAR)*(OutageCoopB));
    
    if(OutageMC<outageAlvo)
        EficienciaCalculada=(R*theta(IndiceTheta)*(1-OutageMC))/Pt;
    else
        EficienciaCalculada=0;
    end 

    EficienciaCSIRC_AlocacaoPotencia_MC(IndiceDistancia)=EficienciaCalculada;
end 

EficienciaCSIRC_AlocacaoTaxaPotencia=[];
for IndiceDistancia=1:length(dre)
    disp(['CSI-RC - d_{RE}=' num2str(dre(IndiceDistancia))])
    
    EficienciaCSIRC1_Theta=[];
    EficienciaCSIRC2_Theta=[];
    for(IndiceTheta=1:length(theta))
        EficienciaCSIRC1_Ps=[];
        EficienciaCSIRC2_Ps=[];
        for(IndicePs=1:length(Ps))
            EficienciaCSIRC1_Pr=[];
            EficienciaCSIRC2_Pr=[];
            for(IndicePr=1:length(Pr))
                gSD=Ps(IndicePs)*ksd/N;
                gSR=Ps(IndicePs)*ksr/N;
                gRD=Pr(IndicePr)*krd/N;
                gSE=Ps(IndicePs)*kse(IndiceDistancia)/N;
                gRE=Pr(IndicePr)*kre(IndiceDistancia)/N;

            %Caso 1:
                gSDi=gSD*hsd_tas;
                gSRi=gSR.*(hsr(1,1,:).^2+hsr(1,2,:).^2);
                gRDi=gRD*hrd_tas;
                gSEi=gSE*(hse(1,1,:).^2 + hse(1,2,:).^2);
                gREi=gRE*(hre(1,1,:).^2 + hre(1,2,:).^2);

                Cb_SDF=log2(1+2*gSDi);
                Ce_SDF=log2(1+2*gSEi);
                outDireto=sum(Cb_SDF-Ce_SDF<2*theta(IndiceTheta)*R)/amostras;

                Cb_SDF=log2(1+gSRi);
                Ce_SDF=log2(1+gREi+gSEi);
                outCooperativoAR=sum(Cb_SDF-Ce_SDF<2*theta(IndiceTheta)*R)/amostras;

                Cb_SDF=log2(1+gSDi+gRDi);
                Ce_SDF=log2(1+gREi+gSEi);
                outCooperativoB=sum(Cb_SDF-Ce_SDF<2*theta(IndiceTheta)*R)/amostras;

                OutageEscolhida=outDireto*(outCooperativoAR+(1-outCooperativoAR)*(outCooperativoB)); 

                PCooperativo=gSR/(gSD + gSR);
                Pt=(1+eff)*Ps(IndicePs)+P_TX(2*1)+P_RX(2*2)+(1-PCooperativo)*(1+eff)*Ps(IndicePs)+(PCooperativo)*((1+eff)*Pr(IndicePr)+P_RX(1*2));

                if(OutageEscolhida<outageAlvo)
                    EficienciaCalculada=(R*theta(IndiceTheta)*(1-OutageEscolhida))/Pt;
                else
                    EficienciaCalculada=0;
                end
                
                EficienciaCSIRC1_Pr = [EficienciaCSIRC1_Pr EficienciaCalculada]; 
                
                %Caso 2
                
                Cb_SDF=log2(1+2*gSDi);
                Ce_SDF=log2(1+gREi+gSEi);
                outDireto=sum(Cb_SDF-Ce_SDF<2*theta(IndiceTheta)*R)/amostras;

                Cb_SDF=log2(1+gSRi);
                Ce_SDF=log2(1+gREi+gSEi);
                outCooperativoAR=sum(Cb_SDF-Ce_SDF<2*theta(IndiceTheta)*R)/amostras;

                Cb_SDF=log2(1+gSDi+gRDi);
                Ce_SDF=log2(1+gREi+gSEi);
                outCooperativoB=sum(Cb_SDF-Ce_SDF<2*theta(IndiceTheta)*R)/amostras;

                OutageEscolhida=outDireto*(outCooperativoAR+(1-outCooperativoAR)*(outCooperativoB)); 

                PCooperativo=gSR/(gSD + gSR);
                Pt=(1+eff)*Ps(IndicePs)+P_TX(2*1)+P_RX(2*2)+(1-PCooperativo)*(1+eff)*Ps(IndicePs)+(PCooperativo)*((1+eff)*Pr(IndicePr)+P_RX(1*2));

                if(OutageEscolhida<outageAlvo)
                    EficienciaCalculada2=(R*theta(IndiceTheta)*(1-OutageEscolhida))/Pt;
                else
                    EficienciaCalculada2=0;
                end

                EficienciaCSIRC2_Pr = [EficienciaCSIRC2_Pr EficienciaCalculada2];  
            end
            [val index]=max(EficienciaCSIRC1_Pr);
            EficienciaCSIRC1_Ps=[EficienciaCSIRC1_Ps val];
            
            [val index]=max(EficienciaCSIRC2_Pr);
            EficienciaCSIRC2_Ps=[EficienciaCSIRC2_Ps val];
            
            ValoresPr2(IndicePs)=Pr(index);
        end    
        [val index]=max(EficienciaCSIRC1_Ps);
        EficienciaCSIRC1_Theta=[EficienciaCSIRC1_Theta val];
        
        [val index]=max(EficienciaCSIRC2_Ps);
        EficienciaCSIRC2_Theta=[EficienciaCSIRC2_Theta val];
        
        ValoresPr(IndiceTheta)=ValoresPr2(index);
        ValoresPs(IndiceTheta)=Ps(index);
    end 
    [val index]=max(EficienciaCSIRC1_Theta);
    EficienciaCSIRC_AlocacaoTaxaPotencia=[EficienciaCSIRC_AlocacaoTaxaPotencia val];
    
    [val index]=max(EficienciaCSIRC2_Theta);
    
    ValorPr=ValoresPr(index);
    ValorPs=ValoresPs(index);
    IndiceTheta=index;
    
    gSD=ValorPs*ksd/N;
    gSR=ValorPs*ksr/N;
    gRD=ValorPr*krd/N;
    gSE=ValorPs*kse(IndiceDistancia)/N;
    gRE=ValorPr*kre(IndiceDistancia)/N;
    
    PCooperativo=gSR/(gSD + gSR);
    Pt=(1+eff)*ValorPs+P_TX(2*1)+P_RX(2*2)+(1-PCooperativo)*(1+eff)*ValorPs+(PCooperativo)*((1+eff)*ValorPr+P_RX(1*2));
    
    Rs=theta(IndiceTheta)*R;
    Ns = 2;
    Nr = 2;
    Nd = 2;
    Ne = 2;

    alfa=((2^(-2*Rs))-1);
    beta=(2*(2^(-2*Rs)));
    S0=0;
    for(k=0:Ns-1)
        S1=0;
        for(iu1=0:k)
            Prob1=((factorial(Ne-1))/((gSE^(-Ne))))*(factorial(iu1+Nd-1)*(((k+1)/gSD)^(-(iu1+Nd))));
            
            S2=0;
            for(m=0:Ne-1)
                S3=0;
                for(v=0:m)
                   S3=S3+nchoosek(m, v)*(alfa^(m-v))*(beta^v)*factorial(v+iu1+Nd-1)*((((k+1)/(gSD))+(beta/gSE))^(-(v+iu1+Nd)));
                end
                S2=S2+((factorial(Ne-1))/((factorial(m))*(gSE^(m-Ne))))*S3;
            end    
            Prob2=exp(-(alfa/gSE))*S2;
            
            S2=0;
            for(w=0:Ne-1)
                S3=0;
                for(v=0:w)
                    S4=0;
                    for(z=0:w-v)
                       Prob3=((factorial(v+Ne-1))/(((1/gSE)-(1/gRE))^(v+Ne)))*(factorial(z+iu1+Nd-1)*((((k+1)/gSD)+(beta/gRE))^(-(z+iu1+Nd))));
                       
                       S5=0;
                       for(m=0:v+Ne-1)
                           S6=0;
                           for(y=0:m)
                               S6=S6+nchoosek(m, y)*(alfa^(m-y))*(beta^y)*factorial(y+z+iu1+Nd-1)*((((k+1)/gSD)+(beta/gSE))^(-(y+z+iu1+Nd)));
                           end
                           S5=S5+(factorial(v+Ne-1)/((factorial(m))*(((1/gSE)-(1/gRE))^(v+Ne-m))))*S6; 
                       end
                       Prob4=exp(-alfa*((1/gSE)-(1/gRE)))*S5;
                       
                       S4=S4+nchoosek(w-v, z)*(alfa^(w-v-z))*(beta^z)*(Prob3-Prob4);
                    end
                    S3=S3+nchoosek(w, v)*((-1)^v)*S4;
                end
                S2=S2+(1/factorial(w))*((1/gRE)^w)*exp(-(alfa/gRE))*S3; 
            end
            ProbSoma=S2;
            
            S1=S1+nchoosek(k, iu1)*((1/factorial(1))^(iu1-0))*((1/(gSD))^(iu1))*(Prob1-Prob2-ProbSoma);
        end
        S0=S0+nchoosek(Ns-1, k)*((-1)^k)*S1;
    end
    OutageDireto=1-(Ns/((gamma(Nd))*(gamma(Ne))*(gSD^Nd)*(gSE^Ne)))*S0;
    
    S0=0;
    for(k=0:Ns-1)
        ni=[k zeros(1,Nd)];
        P0=1;
        for(u=1:Nd-1)
            S1=0;
            for(i_u=0:ni(u))
                ni(u+1)=i_u;
                ValorN=0;
                for(LoopPhi=2:Nd)
                    ValorN=ValorN+ni(LoopPhi);
                end
                
                S2=0;
                for(m=0:Nr-1)
                    ni1=[m zeros(1,Nd)];
                    P1=1;
                    for(u1=1:Nd-1)
                        S3=0;
                        for(i_u1=0:ni1(u1))
                            ni1(u1+1)=i_u1;
                            SomaM=0;
                            for(LoopPhi=2:Nd)
                                SomaM=SomaM+ni1(LoopPhi);
                            end
                            
                            S4=0;
                            for(l=0:Nd+ValorN-1)
                                u=Nd+ValorN-l;
                                v=Nd+SomaM+l;
                                Beta=((k+1)/gSD);
                                alfa=(((m+1)/(gRD))-((k+1)/(gSD)));
                                Prob1=(factorial(Ne-1)/(gSE^(-Ne)))*(((alfa^v)*gamma(u+v))/(v*(alfa+Beta)^(u+v)))*hypergeom([1 u+v], v+1, (alfa)/(alfa+Beta));
                                
                                S5=0;
                                for(p=0:Ne-1)
                                    S6=0;
                                    for(s=0:p)
                                        u=s+Nd+ValorN-l;
                                        v=Nd+SomaM+l;
                                        Beta=((k+1)/gSD)+((2^(-2*Rs))/gSE);
                                        alfa=(((m+1)/(gRD))-((k+1)/(gSD)));
                                        S6=S6+nchoosek(p, s)*(2^(-2*Rs*s))*((2^(-2*Rs)-1)^(p-s))*(((alfa^v)*gamma(u+v))/(v*(alfa+Beta)^(u+v)))*hypergeom([1 u+v], v+1, (alfa)/(alfa+Beta));
                                    end
                                    S5=S5+(factorial(Ne-1)/factorial(p))*(1/(gSE^(p-Ne)))*exp(-(2^(-2*Rs)-1)/(gSE))*S6;
                                end
                                Prob2=S5;
                                
                                S5=0;
                                for(w=0:Ne-1)
                                    S6=0;
                                    for(o=0:w)
                                        S7=0;
                                        for(s=0:w-o)
                                            u=s+Nd+ValorN-l;
                                            v=Nd+SomaM+l;
                                            Beta=((k+1)/gSD)+((2^(-2*Rs))/gRE);
                                            alfa=(((m+1)/(gRD))-((k+1)/(gSD)));
                                            S7=S7+nchoosek(w-o, s)*(2^(-2*Rs*s))*((2^(-2*Rs)-1)^(w-o-s))*(((alfa^v)*gamma(u+v))/(v*(alfa+Beta)^(u+v)))*hypergeom([1 u+v], v+1, (alfa)/(alfa+Beta));
                                        end
                                        
                                        S10=0;
                                        for(t=0:o+Ne-1)
                                            S11=0;
                                            for(x=0:w+t-o)
                                                u2=x+Nd+ValorN-l;
                                                v2=Nd+SomaM+l;
                                                Beta2=((k+1)/gSD)+((2^(-2*Rs))/gSE);
                                                alfa2=(((m+1)/(gRD))-((k+1)/(gSD)));
                                                S11=S11+nchoosek(w+t-o, x)*(2^(-2*Rs*x))*((2^(-2*Rs)-1)^(w+t-o-x))*(((alfa2^v2)*gamma(u2+v2))/(v2*(alfa2+Beta2)^(u2+v2)))*hypergeom([1 u2+v2], v2+1, (alfa2)/(alfa2+Beta2));
                                            end
                                            S10=S10+(factorial(o+Ne-1)/factorial(t))*(((1/gSE)-(1/gRE))^(-(o+Ne-t)))*exp(-(2^(-2*Rs)-1)/(gSE))*S11;
                                        end
                                        
                                        S6=S6+nchoosek(w, o)*((-1)^o)*(((factorial(o+Ne-1))/(((1/gSE)-(1/gRE))^(o+Ne)))*(exp(-(2^(-2*Rs)-1)/(gRE))*S7)-S10);
                                    end
                                    S5=S5+(1/factorial(w))*((1/gRE)^w)*S6;
                                end
                                Prob3=S5;
                                
                                S4=S4+nchoosek(Nd+ValorN-1,l)*((-1)^l)*(alfa^(-(Nd+SomaM+l)))*(Prob1-Prob2-Prob3);
                            end
                            
                            S3=S3+nchoosek(ni1(1), i_u1)*((1/factorial(u1))^(i_u1-ni1(2)))*((1/gRD)^i_u1)*S4;
                        end
                        P1=P1*S3;
                    end
                    S2=S2+nchoosek(Nr-1, m)*((-1)^m)*P1;
                end
                S1=S1+nchoosek(ni(1), i_u)*((1/factorial(u))^(i_u-ni(2)))*((1/gSD)^i_u)*S2;
            end   
            P0=P0*S1;
        end
        S0=S0+nchoosek(Ns-1, k)*((-1)^k)*P0;
    end
    OutageCoopB=1-((Nr*Ns)/((gSE^Ne)*gamma(Ne)*(gamma(Nd)^2)*((gSD*gRD)^Nd)))*S0;
    
    Prob1=0;
    Prob2=0;
    Prob3=0;
    Prob4=0;
    ProbTotal=0;
    Prob1=((factorial(Ne-1)*factorial(Nr-1))/((gSE^(-Ne))*(gSR^(-Nr))));
    
    S0=0;
    for(m=0:Ne-1)
        S1=0;
        for(p=0:m)
           S1=S1+nchoosek(m,p)*(2^(-2*Rs*p))*(((2^(-2*Rs))-1)^(m-p))*factorial(Nr+p-1)*(((1/gSR)+((2^(-2*Rs))/(gSE)))^(-(Nr+p)));
        end
        S0=S0+((factorial(Ne-1))/(factorial(m)))*((1/gSE)^(m-Ne))*exp(-((2^(-2*Rs))-1)/gSE)*S1;
    end
    Prob2=S0;
  
    S0=0;
    for(w=0:Ne-1)
        S1=0;
        for(k=0:w)
            S2=0;
            for(p=0:w-k)
               S2=S2+nchoosek(w-k, p)*(2^(-2*Rs*p))*(((2^(-2*Rs))-1)^(w-k-p))*factorial(Nr+p-1)*((1/gSR)+((2^(-2*Rs))/(gRE)))^(-(Nr+p));
            end
            Prob3=((factorial(k+Ne-1))/(((1/gSE)-(1/gRE))^(k+Ne)))*(exp(-((2^(-2*Rs))-1)/(gRE)))*S2;
            
            S2=0;
            for(o=0:k+Ne-1)
                S3=0;
                for(p=0:w-k+o)
                    S3=S3+nchoosek(w-k+o, p)*(2^(-2*Rs*p))*(((2^(-2*Rs))-1)^(w-k+o-p))*factorial(Nr+p-1)*((1/gSR)+((2^(-2*Rs))/(gSE)))^(-(Nr+p));
                end
                S2=S2+((factorial(k+Ne-1))/(factorial(o)*(((1/gSE)-(1/gRE))^(k+Ne-o))))*exp(-((2^(-2*Rs))-1)/(gSE))*S3;
            end
            Prob4=S2;
            
            S1=S1+nchoosek(w,k)*((-1)^k)*(Prob3-Prob4);
        end
        S0=S0+(1/factorial(w))*((1/gRE)^w)*S1; 
    end
    ProbTotal=S0;
    
    OutageCoopAR=1-(1/(gamma(Ne)*gamma(Nr)*((gSE)^(Ne))*((gSR)^(Nr))))*(Prob1-Prob2-ProbTotal);      
 
    OutageMC=OutageDireto*(OutageCoopAR+(1-OutageCoopAR)*(OutageCoopB));
    
    if(OutageMC<outageAlvo)
        EficienciaCalculada=(R*theta(IndiceTheta)*(1-OutageMC))/Pt;
    else
        EficienciaCalculada=0;
    end 

    EficienciaCSIRC_AlocacaoTaxaPotencia_MC(IndiceDistancia)=EficienciaCalculada;
end 

%%

%Caso 1:
 hAB = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));
        abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));];

hAB_TAS = max([hAB(1,1,:).^2 hAB(2,1,:).^2]);

hAE = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));];

hAE_MRC=(hAE(1,1,:).^2 + hAE(1,2,:).^2);

for IndiceDistancia=1:length(dre)
    disp(['CJ - d_{RE}=' num2str(dre(IndiceDistancia))])
    IndiceTheta=2;
    IndicePs=2;
    IndicePr=2;
    
    gSD=Ps(IndicePs)*ksd/N;
    gSR=Ps(IndicePs)*ksr/N;
    gRD=Pr(IndicePr)*krd/N;
    gSE=Ps(IndicePs)*kse(IndiceDistancia)/N;
    gRE=Pr(IndicePr)*kre(IndiceDistancia)/N;

    %Caso 1:
    gammaI=gRE/1;

    Cb = log2(1 + 2*gSD*hAB_TAS);            

    gI=gammaI.*gamrnd(1,1,1,1,amostras);

    Ce=log2(1 + 2*((gSE.*hAE_MRC)./gI));

    Outage_CJ_MC=sum(Cb-Ce<2*R*theta(IndiceTheta))/amostras;

    Pt=(1+eff)*2*Ps(IndicePs)+P_TX(2*1+2*2)+P_RX(2*1)+(1+eff)*2*Pr(IndicePr);

    if(Outage_CJ_MC<outageAlvo)
        EficienciaCalculada=(R*theta(IndiceTheta)*(1-Outage_CJ_MC))/Pt;
    else
        EficienciaCalculada=0;
    end

    EficienciaCJ(IndiceDistancia)=EficienciaCalculada;
    
    S0=0;
    for(n1=1:Ns)
        S1=0;
        for(n2=0:n1)
            S2=0;
            ni=[n1 n2 0];
            ValorBeta=0;
            for(m=0:Nd-1)
               ValorBeta=ValorBeta+m*(ni(m+1)-ni(m+2));
            end
            for(u=0:Ne-1)
                S3=0;
                for(p=0:ValorBeta)
                    Teta1=M*gamma(p+u+1)*kummerU(p+u+1, p-M+1, ((n1*gSE*(2^(2*R*theta(IndiceTheta))))/(gammaI*gSD)));
                    if(u==0)
                       Teta2=0; 
                    else
                       Teta2=u*gamma(p+u)*kummerU(p+u, p-M, ((n1*gSE*(2^(2*R*theta(IndiceTheta))))/(gammaI*gSD)));
                    end
                    S3=S3+nchoosek(ValorBeta, p)*((gSE/gammaI)^p)*(((2^(2*R*theta(IndiceTheta)))-1)^(ValorBeta-p))*(2^(2*R*theta(IndiceTheta)*p))*exp(-((n1*((2^(2*R*theta(IndiceTheta)))-1))/(gSD)))*(Teta1-Teta2);
                end
                S2=S2+((((-1)^(n1+1))*(gamma(u+M)))/((factorial(u))*((factorial(M-1))*(gSD^ValorBeta))))*S3;
            end
            S1=S1+nchoosek(n1, n2)*((1/factorial(1))^(n1-n2))*S2;
        end
        S0=S0+nchoosek(Ns, n1)*S1;
    end
    Pout=1-S0;
    
    if(Pout<outageAlvo)
        EficienciaCalculada=(R*theta(IndiceTheta)*(1-Pout))/Pt;
    else
        EficienciaCalculada=0;
    end
    
    EficienciaCJ_MC(IndiceDistancia)=EficienciaCalculada;
end 

for IndiceDistancia=1:length(dre)
    disp(['CJ - d_{RE}=' num2str(dre(IndiceDistancia))])
    IndiceTheta=2;
    
    EficienciaCJ1_Ps=[];
    for(IndicePs=1:length(Ps))
        EficienciaCJ1_Pr=[];
        for(IndicePr=1:length(Pr))
            gSD=Ps(IndicePs)*ksd/N;
            gSR=Ps(IndicePs)*ksr/N;
            gRD=Pr(IndicePr)*krd/N;
            gSE=Ps(IndicePs)*kse(IndiceDistancia)/N;
            gRE=Pr(IndicePr)*kre(IndiceDistancia)/N;

            %Caso 1:
            gammaI=gRE/1;

            Cb = log2(1 + 2*gSD*hAB_TAS);            

            gI=gammaI.*gamrnd(1,1,1,1,amostras);

            Ce=log2(1 + 2*((gSE.*hAE_MRC)./gI));

            Outage_CJ_MC=sum(Cb-Ce<2*R*theta(IndiceTheta))/amostras;

            Pt=(1+eff)*2*Ps(IndicePs)+P_TX(2*1+2*2)+P_RX(2*1)+(1+eff)*2*Pr(IndicePr);

            if(Outage_CJ_MC<outageAlvo)
                EficienciaCalculada=(R*theta(IndiceTheta)*(1-Outage_CJ_MC))/Pt;
            else
                EficienciaCalculada=0;
            end

            EficienciaCJ1_Pr = [EficienciaCJ1_Pr EficienciaCalculada];
        end
        [val index]=max(EficienciaCJ1_Pr);
        EficienciaCJ1_Ps=[EficienciaCJ1_Ps val];
        
        ValoresPr(IndicePs)=Pr(index);
    end    
    [val index]=max(EficienciaCJ1_Ps);
    EficienciaCJ_AlocacaoPotencia(IndiceDistancia)=val;
    
    ValorPr=ValoresPr(index);
    ValorPs=Ps(index);
    
    gSD=ValorPs*ksd/N;
    gSR=ValorPs*ksr/N;
    gRD=ValorPr*krd/N;
    gSE=ValorPs*kse(IndiceDistancia)/N;
    gRE=ValorPr*kre(IndiceDistancia)/N;
    
    Pt=(1+eff)*2*ValorPs+P_TX(2*1+2*2)+P_RX(2*1)+(1+eff)*2*ValorPr;

    gammaI=gRE/1;
    
    S0=0;
    for(n1=1:Ns)
        S1=0;
        for(n2=0:n1)
            S2=0;
            ni=[n1 n2 0];
            ValorBeta=0;
            for(m=0:Nd-1)
               ValorBeta=ValorBeta+m*(ni(m+1)-ni(m+2));
            end
            for(u=0:Ne-1)
                S3=0;
                for(p=0:ValorBeta)
                    Teta1=M*gamma(p+u+1)*kummerU(p+u+1, p-M+1, ((n1*gSE*(2^(2*R*theta(IndiceTheta))))/(gammaI*gSD)));
                    if(u==0)
                       Teta2=0; 
                    else
                       Teta2=u*gamma(p+u)*kummerU(p+u, p-M, ((n1*gSE*(2^(2*R*theta(IndiceTheta))))/(gammaI*gSD)));
                    end
                    S3=S3+nchoosek(ValorBeta, p)*((gSE/gammaI)^p)*(((2^(2*R*theta(IndiceTheta)))-1)^(ValorBeta-p))*(2^(2*R*theta(IndiceTheta)*p))*exp(-((n1*((2^(R*theta(IndiceTheta)))-1))/(gSD)))*(Teta1-Teta2);
                end
                S2=S2+((((-1)^(n1+1))*(gamma(u+M)))/((factorial(u))*((factorial(M-1))*(gSD^ValorBeta))))*S3;
            end
            S1=S1+nchoosek(n1, n2)*((1/factorial(1))^(n1-n2))*S2;
        end
        S0=S0+nchoosek(Ns, n1)*S1;
    end
    Pout=1-S0;
    
    if(Pout<outageAlvo)
        EficienciaCalculada=(R*theta(IndiceTheta)*(1-Pout))/Pt;
    else
        EficienciaCalculada=0;
    end
    
    EficienciaCJ_AlocacaoPotencia_MC(IndiceDistancia)=EficienciaCalculada;
end 

EficienciaCJ_AlocacaoTaxaPotencia=[];
for IndiceDistancia=1:length(dre)
    disp(['CJ - d_{RE}=' num2str(dre(IndiceDistancia))])
    
    EficienciaCJ1_Theta=[];
    for(IndiceTheta=1:length(theta))
        EficienciaCJ1_Ps=[];
        for(IndicePs=1:length(Ps))
            EficienciaCJ1_Pr=[];
            for(IndicePr=1:length(Pr))
                gSD=Ps(IndicePs)*ksd/N;
                gSR=Ps(IndicePs)*ksr/N;
                gRD=Pr(IndicePr)*krd/N;
                gSE=Ps(IndicePs)*kse(IndiceDistancia)/N;
                gRE=Pr(IndicePr)*kre(IndiceDistancia)/N;

                %Caso 1:
                gammaI=gRE/1;

                Cb = log2(1 + 2*gSD*hAB_TAS);            

                gI=gammaI.*gamrnd(1,1,1,1,amostras);

                Ce=log2(1 + 2*((gSE.*hAE_MRC)./gI));

                Outage_CJ_MC=sum(Cb-Ce<2*R*theta(IndiceTheta))/amostras;

                Pt=(1+eff)*2*Ps(IndicePs)+P_TX(2*1+2*2)+P_RX(2*1)+(1+eff)*2*Pr(IndicePr);

                if(Outage_CJ_MC<outageAlvo)
                    EficienciaCalculada=(R*theta(IndiceTheta)*(1-Outage_CJ_MC))/Pt;
                else
                    EficienciaCalculada=0;
                end

                EficienciaCJ1_Pr = [EficienciaCJ1_Pr EficienciaCalculada];
            end
            [val index]=max(EficienciaCJ1_Pr);
            EficienciaCJ1_Ps=[EficienciaCJ1_Ps val];
            
            ValoresPr2(IndicePs)=Pr(index);
        end    
        [val , index]=max(EficienciaCJ1_Ps);
        EficienciaCJ1_Theta=[EficienciaCJ1_Theta val];
        
        ValoresPr(IndiceTheta)=ValoresPr2(index);
        ValoresPs(IndiceTheta)=Ps(index);
    end
    [val , index]=max(EficienciaCJ1_Theta);
    EficienciaCJ_AlocacaoTaxaPotencia=[EficienciaCJ_AlocacaoTaxaPotencia val];
    
    ValorPr=ValoresPr(index);
    ValorPs=ValoresPs(index);
    IndiceTheta=index;
    
    gSD=ValorPs*ksd/N;
    gSR=ValorPs*ksr/N;
    gRD=ValorPr*krd/N;
    gSE=ValorPs*kse(IndiceDistancia)/N;
    gRE=ValorPr*kre(IndiceDistancia)/N;
    
    Pt=(1+eff)*2*ValorPs+P_TX(2*1+2*2)+P_RX(2*1)+(1+eff)*2*ValorPr;

    gammaI=gRE/1;
    
    S0=0;
    for(n1=1:Ns)
        S1=0;
        for(n2=0:n1)
            S2=0;
            ni=[n1 n2 0];
            ValorBeta=0;
            for(m=0:Nd-1)
               ValorBeta=ValorBeta+m*(ni(m+1)-ni(m+2));
            end
            for(u=0:Ne-1)
                S3=0;
                for(p=0:ValorBeta)
                    Teta1=M*gamma(p+u+1)*kummerU(p+u+1, p-M+1, ((n1*gSE*(2^(2*R*theta(IndiceTheta))))/(gammaI*gSD)));
                    if(u==0)
                       Teta2=0; 
                    else
                       Teta2=u*gamma(p+u)*kummerU(p+u, p-M, ((n1*gSE*(2^(2*R*theta(IndiceTheta))))/(gammaI*gSD)));
                    end
                    S3=S3+nchoosek(ValorBeta, p)*((gSE/gammaI)^p)*(((2^(2*R*theta(IndiceTheta)))-1)^(ValorBeta-p))*(2^(2*R*theta(IndiceTheta)*p))*exp(-((n1*((2^(R*theta(IndiceTheta)))-1))/(gSD)))*(Teta1-Teta2);
                end
                S2=S2+((((-1)^(n1+1))*(gamma(u+M)))/((factorial(u))*((factorial(M-1))*(gSD^ValorBeta))))*S3;
            end
            S1=S1+nchoosek(n1, n2)*((1/factorial(1))^(n1-n2))*S2;
        end
        S0=S0+nchoosek(Ns, n1)*S1;
    end
    Pout=1-S0;
    
    if(Pout<outageAlvo)
        EficienciaCalculada=(R*theta(IndiceTheta)*(1-Pout))/Pt;
    else
        EficienciaCalculada=0;
    end
%     
    EficienciaCJ_AlocacaoTaxaPotencia_MC(IndiceDistancia)=EficienciaCalculada;
end 

%%
figure(1);
plot(dre, EficienciaCSIRC, 'b*', 'LineWidth', 2, 'MarkerSize', 9);
hold on;

plot(dre, EficienciaCSIRC_AlocacaoPotencia, 'bo', 'LineWidth', 2, 'MarkerSize', 9);
plot(dre, EficienciaCSIRC_AlocacaoTaxaPotencia, 'bd', 'LineWidth', 2, 'MarkerSize', 9);
plot(dre, EficienciaCJ, 'r*', 'LineWidth', 2, 'MarkerSize', 9);
plot(dre, EficienciaCJ_AlocacaoPotencia, 'ro', 'LineWidth', 2, 'MarkerSize', 9);
plot(dre, EficienciaCJ_AlocacaoTaxaPotencia, 'rd', 'LineWidth', 2, 'MarkerSize', 9);

plot(dre, EficienciaCSIRC_MC, 'b--', 'LineWidth', 2, 'MarkerSize', 9);
plot(dre, EficienciaCJ_MC, 'r--', 'LineWidth', 2, 'MarkerSize', 9);
plot(dre, EficienciaCSIRC_AlocacaoPotencia, 'b--', 'LineWidth', 2, 'MarkerSize', 9);
plot(dre, EficienciaCJ_AlocacaoPotencia_MC, 'r--', 'LineWidth', 2, 'MarkerSize', 9);
plot(dre, EficienciaCSIRC_AlocacaoTaxaPotencia_MC, 'b--', 'LineWidth', 2, 'MarkerSize', 9);
plot(dre, EficienciaCJ_AlocacaoTaxaPotencia, 'r--', 'LineWidth', 2, 'MarkerSize', 9);

grid on;
    xlabel('d_{RE} (m)','fontsize',13);
ylabel('$\eta_{s} \rm\bf(bits\,seguros/J/Hz)$','Interpreter','LaTeX','Fontsize',14);
h=legend('CSI-DF - Fixed $P_{R}$, $P_{S}$ and $\mathcal{R}$', ...
         'CSI-DF - Fixed $\mathcal{R}$', ...
         'CSI-DF', ...
         'AN - Fixed $P_{R}$, $P_{S}$ and $\mathcal{R}$', ...
         'AN - Fixed $\mathcal{R}$', ...
         'AN', ...
         'CSI-DF - Monte Carlo', ...
         'AN - Monte Carlo');
set(h,'Interpreter','latex')
xlim([1 301])
