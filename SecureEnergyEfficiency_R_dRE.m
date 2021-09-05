close all;
clear;
clc;

R=3;

xFonte=100;
xDestino=200;
xRelay=150;
xEve=xRelay+25:15:xRelay+300;

dsr=sqrt((xFonte-xRelay).^2);
dsd=sqrt((xFonte-xDestino).^2);
drd=sqrt((xRelay-xDestino).^2);
dse=sqrt((xFonte-xEve).^2);
dre=sqrt((xRelay-xEve).^2);

alpha = 3;           % Path Loss

theta=0.01:0.05:1;

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

Ps_dB = -5:1:5;
Ps = 10.^(Ps_dB./10);

Pr_dB = -5:1:5;
Pr = 10.^(Pr_dB./10);

amostras = 1e5;

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

%Caso CSI-DF:
hsd = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));
       abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));];

hsr = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))];

hrd = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));
       abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))];

hse = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))];

hre = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)))];

hsd_tas = max([hsd(1,1,:).^2 + hsd(1,2,:).^2 hsd(2,1,:).^2 + hsd(2,2,:).^2]);

hrd_tas = max([hrd(1,1,:).^2 + hrd(1,2,:).^2 hrd(2,1,:).^2 + hrd(2,2,:).^2]); 

%Caso CJ:
hAB = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));
        abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));];

hAB_TAS = max([hAB(1,1,:).^2 hAB(2,1,:).^2]);

hAE = [abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras))) abs(sqrt(0.5)*(randn(1, 1, amostras)+1i*randn(1, 1, amostras)));];

hAE_MRC=(hAE(1,1,:).^2 + hAE(1,2,:).^2);

EficienciaAloc_R=[];
EficienciaAlocCJ_R=[];
EficienciaAlocSPC_R=[];
for a=1:length(theta)
    t=theta(a);
    for b=1:length(dre)
        ValorDSR=dsr;
        ValorDRE=dre(b);
        ValorDSE=dse(b);
        ValorDRD=drd;
        ValorDSD=dsd;
        
        ksd = G*lambda^2/((4*pi)^2*ValorDSD.^alpha*Ml*Nf);
        ksr = G*lambda^2/((4*pi)^2*ValorDSR.^alpha*Ml*Nf);
        krd = G*lambda^2/((4*pi)^2*ValorDRD.^alpha*Ml*Nf);
        kre = G*lambda^2/((4*pi)^2*ValorDRE.^alpha*Ml*Nf);
        kse = G*lambda^2/((4*pi)^2*ValorDSE.^alpha*Ml*Nf);
        
        ValoresAlocacaoSDF=[];
        ValoresAlocacaoCJ=[];
        ValoresAlocacaoSPC=[];
        for i=1:length(Ps) 
            for j=1:length(Pr)
                gSD=Ps(i)*ksd/N;
                gSR=Ps(i)*ksr/N;
                gRD=Pr(j)*krd/N;
                gSE=Ps(i)*kse/N;
                gRE=Pr(j)*kre/N;

                gSDi=gSD*hsd_tas;
                gSRi=gSR.*(hsr(1,1,:).^2+hsr(1,2,:).^2);
                gRDi=gRD*hrd_tas;
                gSEi=gSE*(hse(1,1,:).^2 + hse(1,2,:).^2);
                gREi=gRE*(hre(1,1,:).^2 + hre(1,2,:).^2);

                Cb_SDF=log2(1+2*gSDi);
                Ce_SDF=log2(1+gREi+gSEi);
                outDireto=sum(Cb_SDF-Ce_SDF<2*t*R)/amostras;

                Cb_SDF=log2(1+gSRi);
                Ce_SDF=log2(1+gREi+gSEi);
                outCooperativoAR=sum(Cb_SDF-Ce_SDF<2*t*R)/amostras;

                Cb_SDF=log2(1+gSDi+gRDi);
                Ce_SDF=log2(1+gREi+gSEi);
                outCooperativoB=sum(Cb_SDF-Ce_SDF<2*t*R)/amostras;

                OutageMC=outDireto.*(outCooperativoAR+(1-outCooperativoAR).*(outCooperativoB)); 

                PCooperativo=gSR/(gSD + gSR);
                Pt=(1+eff)*Ps(i)+P_TX(2*1)+P_RX(2*2)+(1-PCooperativo)*(1+eff)*Ps(i)+(PCooperativo)*((1+eff)*Pr(j)+P_RX(1*2));
                
                if(OutageMC<outageAlvo)
                    EficienciaCalculada=((R*t*(1-OutageMC))/Pt);
                else
                    EficienciaCalculada=0;
                end

                ValoresAlocacaoSDF = [ValoresAlocacaoSDF; EficienciaCalculada];    
                
                gammaI=gRE/2;

                Cb = log2(1 + 2*gSD*hAB_TAS);            

                gI=gammaI.*(gamrnd(1,1,1,1,amostras)+gamrnd(1,1,1,1,amostras));

                Ce=log2(1 + 2*((gSE.*hAE_MRC)./gI));

                Outage_CJ_MC=sum(Cb-Ce<2*R*t)/amostras;

                Pt=(1+eff)*2*Ps(i)+P_TX(2*1+2*2)+P_RX(2*1)+(1+eff)*2*Pr(j);

                if(Outage_CJ_MC<outageAlvo)
                    EficienciaCalculada=(R*t*(1-Outage_CJ_MC))/Pt;
                else
                    EficienciaCalculada=0;
                end

                ValoresAlocacaoCJ = [ValoresAlocacaoCJ; EficienciaCalculada];
            end
        end
        [val index]=max(ValoresAlocacaoSDF,[],1);
        EficienciaAloc_dre(b)=mean(val);
        [val index]=max(ValoresAlocacaoCJ,[],1);
        EficienciaAlocCJ_dre(b)=mean(val);
        dre(b)
    end
    EficienciaAloc_R=[EficienciaAloc_R;EficienciaAloc_dre];
    EficienciaAlocCJ_R=[EficienciaAlocCJ_R;EficienciaAlocCJ_dre];
    a
end
%%
figure(1)
% set(gcf, 'Units', 'centimeters');
% afFigurePosition = [5 5 16 9];
% set(gcf, 'Position', afFigurePosition);
hSurface =surf(theta*R,dre,EficienciaAloc_R');
set(hSurface,'FaceColor',[0 0 1],'FaceAlpha',1);
hold on;
hSurface =surf(theta*R,dre,EficienciaAlocCJ_R');
set(hSurface,'FaceColor',[1 0 0],'FaceAlpha',1);
xlabel('$\mathcal{R} \rm\bf(secure\,bits/Hz)$','Interpreter','LaTeX','Fontsize',14);
ylabel('$d_{RE} \rm\bf(m)$','Interpreter','LaTeX','Fontsize',14);
zlabel('$\eta_{s} \rm\bf(bits\,seguros/J/Hz)$','Interpreter','LaTeX','Fontsize',14);
legend('CSI-DF', 'AN');
set(gcf, 'Renderer', 'opengl')
ylim([0 300]);