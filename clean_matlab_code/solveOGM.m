function x = solveOGM(M,B,TV,b,x0,options)
% SOLVEOGM Nesterov's optimized-Gradient Method solver
%
%  Function [X] = solveOGM(M,B,TV,b,X0,OPTIONS), solves
%
%  Minimize ||MBx - b||^2  + gammaTV * ||W_TV * TV(Bx)||_q
%
%  where M is the measurement matrix and B is the sparsity basis,
%  TV is the total variation operator and W_TV is the diagonal
%  weight matrices for the sparsity and Total Variation terms
%  respectively.
%
%  To solve the above problem a nonlinear conjugate gradient method
%  as described in the references is used. 
%  Kim et al. IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. 34, NO. 1, JANUARY 2015

%----------------------------------------------------------------------
% Grab input options and set defaults where needed. 
%----------------------------------------------------------------------
gradTol  = options.gradTol;
maxIts   = options.maxIts;
q        = options.qNorm;
mu       = gpuArray(single(options.mu));
gammaTV  = gpuArray(single(options.gammaTV));
weightTV = gpuArray(options.weightTV(:)' );

%----------------------------------------------------------------------
% Initialize local variables.
%----------------------------------------------------------------------
iter   = 0;
x      = x0;
z      = x;
stat   = 0;
t = 1;

% Total-variation and p-Norm terms and flags
fTV    = 0;
gTV    = 0;
flagTV = ((gammaTV ~= 0) && (any(weightTV ~= 0)));
absTV = opAbsDifference([sqrt(length(x0)),sqrt(length(x0))]);%TODO : HACK to set image size
L= computeLipschitz(x);
% Exit conditions (constants).
EXIT_OPTIMAL       = 1;
EXIT_ITERATIONS    = 2;

%----------------------------------------------------------------------
% Log header.
%----------------------------------------------------------------------
logB = ' %5i  %13.7e  %13.7e';
logH = ' %5s  %13s  %13s\n';
disp(sprintf(logH,'Iter','Objective','gNorm'));

% Compute gradient and objective information
[f,g] = computeInfo(x);
dx = g;
gNorm = sqrt(g(:)'*g(:));
gNorm0=gNorm;

%----------------------------------------------------------------------
% MAIN LOOP.
%----------------------------------------------------------------------
while 1
   
     if(isfield(options,'display') && 1 == options.display) 
         
       Ns = sqrt(length(x));
       temp=rot90(real(reshape(x,Ns,Ns)));
       subplot(1,2,1);
       imagesc(temp(Ns/2-Ns/4:Ns/2+Ns/4,Ns/2-Ns/4:Ns/2+Ns/4),[-3 3]);
       axis image;
       colormap(gray);
%       colorbar;
       title(strcat('Iter :',num2str(iter)));
       %drawnow;
       subplot(1,2,2);
       plot(temp(end/2,Ns/2-Ns/4:Ns/2+Ns/4),'r');axis([0,Ns/2,-3,3]);
       title('Line Profile Through Row : 768')
       drawnow;
       %hold on;
       if(1 == iter)
           input('Press any key to continue');
       end
    end
    
   %-------------------------------------------------------------------
   % Test exit conditions.
   %-------------------------------------------------------------------
   
   if (iter  >= maxIts), stat = EXIT_ITERATIONS; end;
   if (gNorm < gradTol), stat = EXIT_OPTIMAL;    end;

   %-------------------------------------------------------------------
   % Print log and act on exit conditions.
   disp(sprintf(logB,iter,f,gNorm./gNorm0));
   
   if stat, break; end % Act on exit conditions.

   %===================================================================
   % Iterations begin here.
   %===================================================================
   iter = iter + 1;

   % Backtracking line-search
   [xNew,zNew,tNew] = ogmUpdate2(x,z,t,dx,L);
   

   % Update x, dx, f, g and gNorm
   x     = xNew;
   [f,g] = computeInfo(x);
   dx = g;
   gNorm = sqrt(g(:)'*g(:));
   z     = zNew;
   t     = tNew;
   
  
   
end

% Print final output.
switch (stat)
   case EXIT_OPTIMAL
      disp(sprintf('\n EXIT -- Optimal solution found\n'));
   case EXIT_ITERATIONS
      disp(sprintf('\n ERROR EXIT -- Too many iterations\n'));
   case EXIT_LINE_ERROR
      disp(sprintf('\n ERROR EXIT -- Linesearch error (%i)\n',lnErr));
   otherwise
      error('Unknown termination condition\n');
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NESTED FUNCTIONS.  These share some vars with workspace above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function [f,g] = computeInfo(x)
        z    = B(x,1);
        Mz   = M(z,1);
        Tz   = TV(z,1);
        Tzw  = Tz.*weightTV;
        xw   = x;
        Tz2w = Tzw.*conj(Tzw);
        x2w  = xw.*conj(xw);
        
        % Compute the objective
        Mzb = Mz - b;
        fRes = Mzb' * Mzb;
        if  flagTV
            fTV = sum(power(Tz2w(:) + mu, q/2));
        end
        
        f = fRes + gammaTV * fTV;
        
        % Compute the gradient
        gRes = 2 * B(M(Mzb,2),2);
        
        if flagTV,
            gTV = q * B(TV(weightTV.*(Tzw.*power(Tz2w + mu,q/2-1)),2),2);
        end;
        g = gRes + gammaTV * gTV;
    end

    function [L]= computeLipschitz(x)
        %L = Lipshcitz constant of the gradient
        temp_x = ones(size(x));
        z    = B(temp_x,1);
        Mz   = M(z,1);
        x_hat = B(M(Mz,2),2); 
        
        Tz   = absTV(z,1);
        Tzw  = Tz.*weightTV;
        Tz2w = Tzw.*conj(Tzw);
        gTV = q * B(absTV(weightTV.*(Tzw.*power(Tz2w + mu,q/2-1)),2),2);
        gTV=gTV.*q*mu^(q/2 - 1);
        
        temp_L = (x_hat+gTV);
        L = (real(temp_L(:)));%max(real(temp_L(:)))   
    end

    function [xNew, zNew,tNew]= ogmUpdate1(x,z,t,grad,L)
        %L = Lipshcitz constant
        zNew = x - grad./L;
        tNew = 0.5*(1+sqrt(1+4*(t^2)));
        xNew =zNew + ((t -1)/tNew)*(zNew-z);
    end

    function [xNew, zNew,tNew]= ogmUpdate2(x,z,t,grad,L)
        %L = Lipshcitz constant
        zNew = x - grad./L;
        tNew = 0.5*(1+sqrt(1+4*(t^2)));
        xNew =zNew + ((t -1)/tNew)*(zNew-z) + (t/tNew)*(zNew-x);
    end


end %solveOGM