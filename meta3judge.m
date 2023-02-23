function res = meta3judge(a,b,c,d)
      if ~isempty(b)
            if length(a) == length(b) && a== b
                  res = c(:);
            else
                  res = d(:);
            end
      else
            res = [];
      end
end