clear all; close all; clc;
%% tSNE
Cs = [2.5, 2.0];%, 2.5,
Ls = [1.2, 0.7];%, 0.5,
load('Simulated_Grid/ODE/pca_mat.mat');
[m,n,rgb] = size(norm_pca.b0_pca);

color_pca = norm_pca;
val_pca = pca;

cross_C = 0.1:0.1:1;
cross_L = 0.1:0.1:1;
curve_C = 1:0.1:3;
curve_L = sqrt(1./curve_C);

% figure(1)
subplot(1,2,1)
him = imshow(color_pca.b0_pca);
set(him,'XData',[0.1, 3.0],'YData',[0.1, 3.0]);
hold on

plot(cross_C,cross_L,'k-','LineWidth',3)
% plot(curve_C,curve_L,'k-','LineWidth',3)
plot(1:0.1:3,(1:0.1:3).*0+1,'k-','LineWidth',3)
plot((1:0.1:3).*0+1,1:0.1:3,'k-','LineWidth',3)
% xline(1,'k-','LineWidth',3)
% yline(1,'k-','LineWidth',3)
scatter(Cs,Ls, 120,...
    'filled', ...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor','k')
ylim([0.1,3.0])
xlim([0.1,3.0])
hold off
set(gca,'YDir','normal')
xlabel('C')
ylabel('L')
% title('A) Betti-0 & Betti-1 colormap by PCA','FontSize',24)
set(gca,'FontSize',18)
axis on
xticks([0.1 1 3])
yticks([0.1 1 3])
xticklabels({'0.1','1','3'})
yticklabels({'0.1','1','3'})
%%
color_pca.b0_pca = reshape(color_pca.b0_pca,[900,3]);
color_pca.b1_pca = reshape(color_pca.b1_pca,[900,3]);
color_pca.b01_pca = reshape(color_pca.b01_pca,[900,3]);

%%
subplot(1,2,2)
hold on
for im = 1:900
        scatter3(val_pca.b0_pca(1,im),...
            val_pca.b0_pca(2,im),...
            val_pca.b0_pca(3,im),...
            'filled',...
            'MarkerEdgeColor',color_pca.b0_pca(im,:),...
            'MarkerFaceColor',color_pca.b0_pca(im,:))
end
hold off
view([49,9])
box on
grid on
% title('B) PCA on Betti-0 & Betti-1','FontSize',24)
set(gca,'FontSize',18)
xlabel('t-SNE Dim. 1')
ylabel('t-SNE Dim. 2')
zlabel('t-SNE Dim. 3')