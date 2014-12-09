% Plot a character outline curve
%
%   plotCharacter(u, varargin)
%
% Parameters:
%       u           Vector (length 1024) containing the character
%                   outline to plot
%       varargin    Specify extra parameters to the plot function to
%                   customise appearance, e.g. 'b-' will be a blue line
%
%
function plotCharacter(u, varargin)

% Use varargin to specify extra parameters to pass to plot()
% If none are specified default to blue line with dot markers..
if (isempty(varargin))
    varargin = {'b.-'};
end

NumOutputReconstructionPoints = 512;

Y = u(:);

assert (mod(numel(Y), NumOutputReconstructionPoints) == 0);

N = numel(Y);
M = floor(N / 2);
assert (N == 2*M);

Y = reshape(Y, [], 2);

Sm = repmat(NumOutputReconstructionPoints, ...
    ceil(M / NumOutputReconstructionPoints), 1);
Sn = 2;

Outline = mat2cell(Y, Sm, Sn);

hold on;

if (iscell(Outline))
    for i = 1:length(Outline)
        plot(Outline{i}(:,1), Outline{i}(:,2), varargin{:});
        % plot(Outline(1,1), Outline(1,2), 'k.', 'MarkerSize', 30);
    end
else
    plot(Outline(:,1), Outline(:,2), varargin{:});
    % plot(Outline(1,1), Outline(1,2), 'k.', 'MarkerSize', 30);
end

hold off;
axis tight;
axis equal;
axis off;

end
