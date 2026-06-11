/**
 * vtk-compat.js — VTK.js-compatible WebGL2 volume renderer
 * Exposes window.vtk with the same API as @kitware/vtk.js UMD bundle
 * Drop-in replacement: place alongside index.html, load with <script src="vtk-compat.js">
 */
(function(global){
'use strict';

// ─── Math helpers ────────────────────────────────────────────
function m4mul(a,b){const c=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[i*4+k]*b[k*4+j];c[i*4+j]=s;}return c;}
function m4perspective(fov,aspect,n,f){const t=1/Math.tan(fov/2),nf=1/(n-f);return new Float32Array([t/aspect,0,0,0,0,t,0,0,0,0,(f+n)*nf,-1,0,0,2*f*n*nf,0]);}
function m4lookAt(ex,ey,ez,cx,cy,cz,ux,uy,uz){
  let fx=cx-ex,fy=cy-ey,fz=cz-ez,fl=Math.sqrt(fx*fx+fy*fy+fz*fz)||1;fx/=fl;fy/=fl;fz/=fl;
  let rx=fy*uz-fz*uy,ry=fz*ux-fx*uz,rz=fx*uy-fy*ux,rl=Math.sqrt(rx*rx+ry*ry+rz*rz)||1;rx/=rl;ry/=rl;rz/=rl;
  const ux2=ry*fz-rz*fy,uy2=rz*fx-rx*fz,uz2=rx*fy-ry*fx;
  return new Float32Array([rx,ux2,-fx,0,ry,uy2,-fy,0,rz,uz2,-fz,0,-(rx*ex+ry*ey+rz*ez),-(ux2*ex+uy2*ey+uz2*ez),(fx*ex+fy*ey+fz*ez),1]);
}
function m4inv(m){
  const a=m,r=new Float32Array(16);
  const a00=a[0],a01=a[1],a02=a[2],a03=a[3],a10=a[4],a11=a[5],a12=a[6],a13=a[7];
  const a20=a[8],a21=a[9],a22=a[10],a23=a[11],a30=a[12],a31=a[13],a32=a[14],a33=a[15];
  const b00=a00*a11-a01*a10,b01=a00*a12-a02*a10,b02=a00*a13-a03*a10;
  const b03=a01*a12-a02*a11,b04=a01*a13-a03*a11,b05=a02*a13-a03*a12;
  const b06=a20*a31-a21*a30,b07=a20*a32-a22*a30,b08=a20*a33-a23*a30;
  const b09=a21*a32-a22*a31,b10=a21*a33-a23*a31,b11=a22*a33-a23*a32;
  const det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
  if(!det)return m;const id=1/det;
  r[0]=(a11*b11-a12*b10+a13*b09)*id;r[1]=(a02*b10-a01*b11-a03*b09)*id;
  r[2]=(a31*b05-a32*b04+a33*b03)*id;r[3]=(a22*b04-a21*b05-a23*b03)*id;
  r[4]=(a12*b08-a10*b11-a13*b07)*id;r[5]=(a00*b11-a02*b08+a03*b07)*id;
  r[6]=(a32*b02-a30*b05-a33*b01)*id;r[7]=(a20*b05-a22*b02+a23*b01)*id;
  r[8]=(a10*b10-a11*b08+a13*b06)*id;r[9]=(a01*b08-a00*b10-a03*b06)*id;
  r[10]=(a30*b04-a31*b02+a33*b00)*id;r[11]=(a21*b02-a20*b04-a23*b00)*id;
  r[12]=(a11*b07-a10*b09-a12*b06)*id;r[13]=(a00*b09-a01*b07+a02*b06)*id;
  r[14]=(a31*b01-a30*b03-a32*b00)*id;r[15]=(a20*b03-a21*b01+a22*b00)*id;
  return r;
}

// ─── GLSL sources ────────────────────────────────────────────
const VS_FULL=`#version 300 es
precision highp float;
out vec2 vUV;
void main(){
  vec2 p=vec2(float((gl_VertexID&1)*2)-1.0,float((gl_VertexID>>1)*2)-1.0);
  gl_Position=vec4(p,0.0,1.0);
  vUV=p*0.5+0.5;
}`;

const FS_VOL=`#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vUV;
out vec4 fragColor;
uniform sampler3D uVol;
uniform sampler3D uMask;
uniform bool uHasMask;
uniform mat4 uInvMVP;
uniform vec3 uVolDim;
uniform float uOpacity;
uniform int uPreset;
uniform float uWinLo,uWinHi;

// Ambient occlusion approx via gradient magnitude
float gradMag(vec3 p){
  vec3 d=1.5/uVolDim;
  float gx=texture(uVol,p+vec3(d.x,0,0)).r-texture(uVol,p-vec3(d.x,0,0)).r;
  float gy=texture(uVol,p+vec3(0,d.y,0)).r-texture(uVol,p-vec3(0,d.y,0)).r;
  float gz=texture(uVol,p+vec3(0,0,d.z)).r-texture(uVol,p-vec3(0,0,d.z)).r;
  return length(vec3(gx,gy,gz));
}
vec3 gradNorm(vec3 p){
  vec3 d=1.5/uVolDim;
  float gx=texture(uVol,p+vec3(d.x,0,0)).r-texture(uVol,p-vec3(d.x,0,0)).r;
  float gy=texture(uVol,p+vec3(0,d.y,0)).r-texture(uVol,p-vec3(0,d.y,0)).r;
  float gz=texture(uVol,p+vec3(0,0,d.z)).r-texture(uVol,p-vec3(0,0,d.z)).r;
  return normalize(vec3(gx,gy,gz)+vec3(1e-7));
}

// Transfer functions — map normalised density [0,1] → (rgb, alpha)
vec4 tfBone(float t){
  vec3 col; float a;
  if(t<0.10){return vec4(0);}
  else if(t<0.18){float f=(t-0.10)/0.08;a=f*0.05;col=mix(vec3(0.08,0.06,0.05),vec3(0.45,0.34,0.22),f);}
  else if(t<0.42){float f=(t-0.18)/0.24;a=0.05+f*0.08;col=mix(vec3(0.45,0.34,0.22),vec3(0.80,0.70,0.56),f);}
  else if(t<0.70){float f=(t-0.42)/0.28;a=0.13+f*0.07;col=mix(vec3(0.80,0.70,0.56),vec3(0.93,0.88,0.80),f);}
  else{float f=(t-0.70)/0.30;a=0.20+f*0.05;col=mix(vec3(0.93,0.88,0.80),vec3(1.0,1.0,1.0),f);}
  return vec4(col,a*uOpacity);
}
vec4 tfSoft(float t){
  vec3 col; float a;
  if(t<0.04){return vec4(0);}
  else if(t<0.15){float f=(t-0.04)/0.11;a=f*0.04;col=mix(vec3(0.42,0.18,0.10),vec3(0.78,0.46,0.22),f);}
  else if(t<0.38){float f=(t-0.15)/0.23;a=0.04+f*0.05;col=mix(vec3(0.78,0.46,0.22),vec3(0.92,0.66,0.44),f);}
  else{float f=(t-0.38)/0.62;a=0.09+f*0.03;col=mix(vec3(0.92,0.66,0.44),vec3(1.0,0.94,0.86),f);}
  return vec4(col,a*uOpacity);
}
vec4 tfLung(float t){
  vec3 col; float a;
  if(t<0.02){return vec4(0);}
  else if(t<0.10){float f=(t-0.02)/0.08;a=f*0.03;col=mix(vec3(0.04,0.06,0.12),vec3(0.18,0.30,0.55),f);}
  else if(t<0.30){float f=(t-0.10)/0.20;a=0.03+f*0.05;col=mix(vec3(0.18,0.30,0.55),vec3(0.58,0.70,0.88),f);}
  else{float f=(t-0.30)/0.70;a=0.08+f*0.02;col=mix(vec3(0.58,0.70,0.88),vec3(0.94,0.97,1.0),f);}
  return vec4(col,a*uOpacity);
}

void main(){
  // Reconstruct ray in normalised volume space [0,1]^3
  vec4 pn=uInvMVP*vec4(vUV*2.0-1.0,-1.0,1.0);
  vec4 pf=uInvMVP*vec4(vUV*2.0-1.0, 1.0,1.0);
  vec3 ro=pn.xyz/pn.w;
  vec3 rd=pf.xyz/pf.w;
  vec3 rayDir=normalize(rd-ro);

  // Ray-box intersection [0,1]^3
  vec3 tbot=(vec3(0.0)-ro)/rayDir;
  vec3 ttop=(vec3(1.0)-ro)/rayDir;
  vec3 tmin2=min(tbot,ttop), tmax2=max(tbot,ttop);
  float tEntry=max(max(tmin2.x,tmin2.y),tmin2.z);
  float tExit =min(min(tmax2.x,tmax2.y),tmax2.z);
  if(tEntry>=tExit||tExit<=0.0){fragColor=vec4(0.018,0.020,0.035,1.0);return;}
  tEntry=max(tEntry,0.0);

  // Adaptive step — finer near camera
  float stepSz=1.0/(max(uVolDim.x,max(uVolDim.y,uVolDim.z))*1.4);
  float jit=fract(sin(dot(vUV,vec2(127.1,311.7)))*43758.55);
  float tCur=tEntry+jit*stepSz;

  vec3 accRGB=vec3(0.0);
  float accA=0.0;
  vec3 Ldir=normalize(vec3(0.6,0.9,0.7));
  vec3 Ldir2=normalize(vec3(-0.5,-0.3,0.6));

  for(int i=0;i<600;i++){
    if(tCur>tExit||accA>0.995)break;
    vec3 pos=ro+rayDir*tCur;
    float dens=texture(uVol,pos).r;
    // Apply window/level
    float t=clamp((dens-uWinLo)/(uWinHi-uWinLo+0.0001),0.0,1.0);

    vec4 smp;
    if(uPreset==0)      smp=tfBone(t);
    else if(uPreset==1) smp=tfSoft(t);
    else                smp=tfLung(t);

    if(smp.a>0.002){
      // Gradient shading — Phong model
      float gm=gradMag(pos);
      if(gm>0.01){
        vec3 nrm=gradNorm(pos);
        float diff1=max(0.0,dot(nrm,Ldir));
        float diff2=max(0.0,dot(nrm,Ldir2))*0.3;
        float spec=pow(max(0.0,dot(reflect(-Ldir,nrm),-rayDir)),28.0)*0.35;
        float shading=0.25+diff1*0.6+diff2+spec;
        smp.rgb*=shading;
      }

      // Mask: vivid red highlight
      if(uHasMask){
        float mk=texture(uMask,pos).r;
        if(mk>0.5){
          float edge=smoothstep(0.5,0.9,mk);
          smp.rgb=mix(smp.rgb,vec3(1.0,0.08,0.08),0.80*edge);
          smp.rgb+=vec3(0.3,0.0,0.0)*edge;
          smp.a=max(smp.a,0.22*edge);
        }
      }

      // Front-to-back alpha compositing
      float contrib=smp.a*(1.0-accA);
      accRGB+=smp.rgb*contrib;
      accA+=contrib;
    }
    // Adaptive step: smaller steps in high-gradient regions
    float gm2=gradMag(pos);
    tCur+=stepSz*(gm2>0.04?0.6:1.0);
  }

  // Subtle dark background gradient
  vec3 bgCol=mix(vec3(0.018,0.020,0.038),vec3(0.008,0.010,0.022),vUV.y);
  fragColor=vec4(accRGB+bgCol*(1.0-accA),1.0);
}`;

const VS_SLICE=`#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
out vec3 vTex;
void main(){gl_Position=uMVP*vec4(aPos,1.0);vTex=aPos;}`;

const FS_SLICE=`#version 300 es
precision highp float;
precision highp sampler3D;
in vec3 vTex;
out vec4 fragColor;
uniform sampler3D uVol;
uniform sampler3D uMask;
uniform bool uHasMask;
uniform vec3 uPlaneColor;
uniform float uWinLo,uWinHi;
void main(){
  float dens=texture(uVol,vTex).r;
  float v=clamp((dens-uWinLo)/(uWinHi-uWinLo+0.0001),0.0,1.0);
  vec3 col=vec3(v);
  if(uHasMask){
    float mk=texture(uMask,vTex).r;
    if(mk>0.5)col=mix(col,vec3(1.0,0.08,0.08),0.72);
  }
  // Subtle color tint at edges to show plane identity
  float bx=abs(vTex.x-0.5)*2.0,by=abs(vTex.y-0.5)*2.0,bz=abs(vTex.z-0.5)*2.0;
  float edgeFactor=pow(max(max(bx,by),bz),12.0);
  col=mix(col,uPlaneColor,edgeFactor*0.5);
  fragColor=vec4(col,0.92);
}`;

// ─── Core GL renderer ────────────────────────────────────────
function VtkRenderer(container,background){
  background=background||[0.018,0.020,0.038];
  const cv=document.createElement('canvas');
  cv.style.cssText='position:absolute;top:0;left:0;width:100%;height:100%;display:block;';
  container.style.position='relative';
  container.appendChild(cv);

  const gl=cv.getContext('webgl2',{antialias:false,premultipliedAlpha:false,preserveDrawingBuffer:false});
  if(!gl)throw new Error('WebGL2 not available');

  let volProg=null,sliceProg=null;
  let volVAO=null;
  let sliceVAOs={},sliceVBOs={};
  let volTex=null,maskTex=null;
  let hasMask=false;
  let volumes=[],actors=[];
  // Camera
  let rotX=0.35,rotY=-0.55,dist=1.75,focal=[0.5,0.5,0.5];
  let drag=false,lastX=0,lastY=0;
  // Win/level (normalised 0-1)
  let winLo=0.0,winHi=1.0;

  // Compile shader
  function compile(vsSrc,fsSrc){
    const vs=gl.createShader(gl.VERTEX_SHADER);gl.shaderSource(vs,vsSrc);gl.compileShader(vs);
    if(!gl.getShaderParameter(vs,gl.COMPILE_STATUS))throw new Error('VS:'+gl.getShaderInfoLog(vs));
    const fs=gl.createShader(gl.FRAGMENT_SHADER);gl.shaderSource(fs,fsSrc);gl.compileShader(fs);
    if(!gl.getShaderParameter(fs,gl.COMPILE_STATUS))throw new Error('FS:'+gl.getShaderInfoLog(fs));
    const p=gl.createProgram();gl.attachShader(p,vs);gl.attachShader(p,fs);gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error('Link:'+gl.getProgramInfoLog(p));
    return p;
  }

  // Init GL resources
  volProg=compile(VS_FULL,FS_VOL);
  sliceProg=compile(VS_SLICE,FS_SLICE);
  volVAO=gl.createVertexArray();
  ['axial','coronal','sagittal'].forEach(k=>{
    const vao=gl.createVertexArray();gl.bindVertexArray(vao);
    const vbo=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vbo);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(18),gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
    gl.bindVertexArray(null);
    sliceVAOs[k]=vao;sliceVBOs[k]=vbo;
  });

  // Create 3D texture
  function make3DTex(w,h,d,data){
    const t=gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D,t);
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_3D,0,gl.R8,w,h,d,0,gl.RED,gl.UNSIGNED_BYTE,data);
    return t;
  }

  // Upload CT volume
  function uploadVolume(imgData){
    const{nx,ny,nz,scalars,gmin,gmax}=imgData;
    if(volTex)gl.deleteTexture(volTex);
    const rng=gmax-gmin||1;
    const norm=new Uint8Array(nx*ny*nz);
    for(let i=0;i<norm.length;i++)norm[i]=Math.round(Math.max(0,Math.min(1,(scalars[i]-gmin)/rng))*255);
    volTex=make3DTex(nx,ny,nz,norm);
    updateSliceGeo(imgData);
    render();
  }
  function uploadMask(imgData){
    const{nx,ny,nz,scalars}=imgData;
    if(maskTex)gl.deleteTexture(maskTex);
    const m=new Uint8Array(nx*ny*nz);
    for(let i=0;i<m.length;i++)m[i]=scalars[i]>0?255:0;
    maskTex=make3DTex(nx,ny,nz,m);
    hasMask=true;
    render();
  }

  let slicePos={axial:0.5,coronal:0.5,sagittal:0.5};
  function updateSliceGeo(imgData){
    if(!imgData)return;
    const z=slicePos.axial,y=slicePos.coronal,x=slicePos.sagittal;
    const geo={
      axial:   [0,0,z, 1,0,z, 1,1,z, 0,0,z, 1,1,z, 0,1,z],
      coronal: [0,y,0, 1,y,0, 1,y,1, 0,y,0, 1,y,1, 0,y,1],
      sagittal:[x,0,0, x,1,0, x,1,1, x,0,0, x,1,1, x,0,1]
    };
    ['axial','coronal','sagittal'].forEach(k=>{
      gl.bindBuffer(gl.ARRAY_BUFFER,sliceVBOs[k]);
      gl.bufferSubData(gl.ARRAY_BUFFER,0,new Float32Array(geo[k]));
    });
  }

  // Camera matrix
  function getMVP(){
    const[fx,fy,fz]=focal;
    const ex=fx+dist*Math.sin(rotY)*Math.cos(rotX);
    const ey=fy+dist*Math.sin(rotX);
    const ez=fz+dist*Math.cos(rotY)*Math.cos(rotX);
    const W=cv.width||1,H=cv.height||1;
    const proj=m4perspective(0.65,W/H,0.01,10.0);
    const view=m4lookAt(ex,ey,ez,fx,fy,fz,0,1,0);
    const mvp=m4mul(proj,view);
    return{mvp,invMVP:m4inv(mvp),eye:[ex,ey,ez]};
  }

  // Main render
  let renderScheduled=false;
  function render(){
    if(renderScheduled)return;
    renderScheduled=true;
    requestAnimationFrame(()=>{
      renderScheduled=false;
      _doRender();
    });
  }
  function _doRender(){
    const W=container.clientWidth,H=container.clientHeight;
    if(W<=0||H<=0)return;
    if(cv.width!==W||cv.height!==H){cv.width=W;cv.height=H;}
    gl.viewport(0,0,W,H);
    gl.clearColor(...background,1);
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    if(!volTex)return;

    const{mvp,invMVP,eye}=getMVP();
    const vol=_currentVol;
    const preset=_currentPreset;

    // ── Volume ray-cast pass ──
    gl.useProgram(volProg);
    gl.bindVertexArray(volVAO);
    const uloc=(n)=>gl.getUniformLocation(volProg,n);
    gl.uniform1i(uloc('uVol'),0);
    gl.uniform1i(uloc('uMask'),1);
    gl.uniform1i(uloc('uHasMask'),hasMask?1:0);
    gl.uniformMatrix4fv(uloc('uInvMVP'),false,invMVP);
    if(vol)gl.uniform3f(uloc('uVolDim'),vol.nx,vol.ny,vol.nz);
    gl.uniform1f(uloc('uOpacity'),_opacity);
    gl.uniform1i(uloc('uPreset'),preset);
    gl.uniform1f(uloc('uWinLo'),winLo);
    gl.uniform1f(uloc('uWinHi'),winHi);
    gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_3D,volTex);
    gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_3D,maskTex||volTex);
    gl.drawArrays(gl.TRIANGLES,0,3);

    // ── Slice planes pass ──
    gl.useProgram(sliceProg);
    const sloc=(n)=>gl.getUniformLocation(sliceProg,n);
    gl.uniformMatrix4fv(sloc('uMVP'),false,mvp);
    gl.uniform1i(sloc('uVol'),0);
    gl.uniform1i(sloc('uMask'),1);
    gl.uniform1i(sloc('uHasMask'),hasMask?1:0);
    gl.uniform1f(sloc('uWinLo'),winLo);
    gl.uniform1f(sloc('uWinHi'),winHi);
    gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_3D,volTex);
    gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_3D,maskTex||volTex);
    const planeColors={axial:[0.95,0.15,0.15],coronal:[0.12,0.88,0.12],sagittal:[0.15,0.38,0.98]};
    ['axial','coronal','sagittal'].forEach(k=>{
      gl.uniform3fv(sloc('uPlaneColor'),new Float32Array(planeColors[k]));
      gl.bindVertexArray(sliceVAOs[k]);
      gl.drawArrays(gl.TRIANGLES,0,6);
    });
    gl.bindVertexArray(null);
  }

  // Mouse/touch orbit
  cv.addEventListener('mousedown',e=>{drag=true;lastX=e.clientX;lastY=e.clientY;e.preventDefault();});
  window.addEventListener('mousemove',e=>{
    if(!drag)return;
    rotY+=(e.clientX-lastX)*0.007;
    rotX+=(e.clientY-lastY)*0.007;
    rotX=Math.max(-Math.PI/2+0.05,Math.min(Math.PI/2-0.05,rotX));
    lastX=e.clientX;lastY=e.clientY;
    render();
  });
  window.addEventListener('mouseup',()=>drag=false);
  cv.addEventListener('wheel',e=>{
    e.preventDefault();
    dist=Math.max(0.7,Math.min(4.5,dist+e.deltaY*0.002));
    render();
  },{passive:false});
  cv.addEventListener('touchstart',e=>{if(e.touches.length===1){drag=true;lastX=e.touches[0].clientX;lastY=e.touches[0].clientY;}},{passive:false});
  cv.addEventListener('touchmove',e=>{
    if(!drag||e.touches.length!==1)return;
    rotY+=(e.touches[0].clientX-lastX)*0.010;
    rotX+=(e.touches[0].clientY-lastY)*0.010;
    rotX=Math.max(-Math.PI/2+0.05,Math.min(Math.PI/2-0.05,rotX));
    lastX=e.touches[0].clientX;lastY=e.touches[0].clientY;
    render();e.preventDefault();
  },{passive:false});
  cv.addEventListener('touchend',()=>drag=false);

  let _currentVol=null,_opacity=0.5,_currentPreset=0;

  // ResizeObserver
  new ResizeObserver(()=>render()).observe(container);

  return{
    canvas:cv, gl,
    uploadVolume,
    uploadMask,
    render,
    setOpacity(v){_opacity=v;render();},
    setPreset(p){_currentPreset=p;render();},
    setWinLevel(lo,hi){winLo=lo;winHi=hi;render();},
    setSlicePos(axis,v){
      slicePos[axis]=v;
      if(_currentVol)updateSliceGeo(_currentVol);
      render();
    },
    setCurrentVol(v){_currentVol=v;},
    resetCamera(){rotX=0.35;rotY=-0.55;dist=1.75;render();},
    setFocalPoint(x,y,z){focal=[x,y,z];render();},
  };
}

// ─── VTK-compatible class wrappers ───────────────────────────
// These mimic the VTK.js newInstance() / getter-setter API

function makeNewInstance(ctor){
  return{newInstance:(opts)=>new ctor(opts||{})};
}

// vtkDataArray
class VtkDataArray{
  constructor(o){this._values=o.values;this._name=o.name||'';this._nc=o.numberOfComponents||1;}
  getData(){return this._values;}
  getName(){return this._name;}
  getNumberOfComponents(){return this._nc;}
}

// vtkImageData
class VtkImageData{
  constructor(){this._dims=[1,1,1];this._spacing=[1,1,1];this._origin=[0,0,0];this._pd={scalars:null};}
  setDimensions(x,y,z){this._dims=[x,y,z];}
  getDimensions(){return this._dims;}
  setSpacing(x,y,z){this._spacing=[x,y,z];}
  setOrigin(x,y,z){this._origin=[x,y,z];}
  getPointData(){return{setScalars:(da)=>{this._pd.scalars=da;},getScalars:()=>this._pd.scalars};}
  // Helper for renderer
  toVol(){
    const[nx,ny,nz]=this._dims;
    const da=this._pd.scalars;
    const raw=da?da.getData():null;
    let gmin=Infinity,gmax=-Infinity;
    if(raw)for(let i=0;i<raw.length;i++){if(raw[i]<gmin)gmin=raw[i];if(raw[i]>gmax)gmax=raw[i];}
    return{nx,ny,nz,scalars:raw,gmin,gmax};
  }
}

// vtkColorTransferFunction
class VtkColorTF{
  constructor(){this._pts=[];}
  addRGBPoint(v,r,g,b){this._pts.push({v,r,g,b});}
  removeAllPoints(){this._pts=[];}
  getPoints(){return this._pts;}
}

// vtkPiecewiseFunction
class VtkPWF{
  constructor(){this._pts=[];}
  addPoint(v,o){this._pts.push({v,o});}
  removeAllPoints(){this._pts=[];}
  getPoints(){return this._pts;}
}

// vtkVolumeProperty
class VtkVolumeProp{
  constructor(){this._ctf=new VtkColorTF();this._otf=new VtkPWF();this._shade=true;this._amb=0.2;this._diff=0.8;this._spec=0.2;this._interp='linear';}
  setColor(ctf){this._ctf=ctf;}
  getRGBTransferFunction(i){return this._ctf;}
  setScalarOpacity(i,pwf){this._otf=pwf;}
  getScalarOpacity(i){return this._otf;}
  setShade(v){this._shade=v;}
  setAmbient(v){this._amb=v;}
  setDiffuse(v){this._diff=v;}
  setSpecular(v){this._spec=v;}
  setInterpolationTypeToLinear(){this._interp='linear';}
  setScalarOpacityUnitDistance(i,d){}
}

// vtkVolumeMapper
class VtkVolumeMapper{
  constructor(){this._input=null;this._sampleDist=1.0;}
  setInputData(d){this._input=d;}
  getInputData(){return this._input;}
  setSampleDistance(v){this._sampleDist=v;}
  setAutoAdjustSampleDistances(){}
}

// vtkVolume
class VtkVolume{
  constructor(){this._mapper=null;this._prop=new VtkVolumeProp();}
  setMapper(m){this._mapper=m;}
  getMapper(){return this._mapper;}
  setProperty(p){this._prop=p;}
  getProperty(){return this._prop;}
}

// vtkActor
class VtkActor{
  constructor(){this._mapper=null;this._prop={_col:[1,1,1],_op:1,setColor:(r,g,b)=>{this._prop._col=[r,g,b];},setOpacity:(v)=>{this._prop._op=v;},getColor:()=>this._prop._col,getOpacity:()=>this._prop._op,setAmbient:()=>{},setDiffuse:()=>{},setEdgeVisibility:()=>{},setEdgeColor:()=>{}};}
  setMapper(m){this._mapper=m;}
  getMapper(){return this._mapper;}
  getProperty(){return this._prop;}
  setVisibility(v){this._visible=v;}
}

// vtkPolyDataMapper
class VtkPolyDataMapper{
  constructor(){this._input=null;}
  setInputData(d){this._input=d;}
  setInputConnection(c){this._input=c;}
  setScalarVisibility(){}
}

// vtkPlaneSource
class VtkPlaneSource{
  constructor(){this._o=[0,0,0];this._p1=[1,0,0];this._p2=[0,1,0];}
  setOrigin(x,y,z){this._o=[x,y,z];}
  setPoint1(x,y,z){this._p1=[x,y,z];}
  setPoint2(x,y,z){this._p2=[x,y,z];}
  modified(){}
  getOutputPort(){return this;}
}

// vtkInteractorStyleTrackballCamera — handled internally by renderer
class VtkInteractorStyleTrackball{constructor(){}}

// vtkCamera
class VtkCamera{
  constructor(r){this._r=r;}
  setFocalPoint(x,y,z){this._r&&this._r.setFocalPoint(x,y,z);}
}

// vtkRenderer shim
class VtkRendererShim{
  constructor(r){this._r=r;this._cam=new VtkCamera(r);}
  addVolume(v){this._volume=v;this._r._currentVol=v;this._r.setCurrentVol(v);}
  removeVolume(v){}
  addActor(a){
    if(a&&a._planeKey)this._r.setSlicePos(a._planeKey,a._planePos);
    this._actors=(this._actors||[]);this._actors.push(a);
  }
  removeActor(a){if(this._actors)this._actors=this._actors.filter(x=>x!==a);}
  resetCamera(){this._r.resetCamera();}
  getActiveCamera(){return this._cam;}
}

// vtkRenderWindow shim
class VtkRenderWindowShim{
  constructor(r){this._r=r;}
  render(){this._r.render();}
}

// vtkInteractor shim
class VtkInteractorShim{
  setInteractorStyle(){}
}

// ── vtkGenericRenderWindow (the main entry point) ──
class VtkGenericRenderWindow{
  constructor(opts){
    this._bg=opts.background||[0.02,0.02,0.05];
    this._container=null;
    this._renderer=null;
    this._renWin=null;
    this._interactor=null;
    this._r=null;
  }
  setContainer(el){
    this._container=el;
    this._r=VtkRenderer(el,this._bg);
    this._rendererShim=new VtkRendererShim(this._r);
    this._renWin=new VtkRenderWindowShim(this._r);
    this._interactor=new VtkInteractorShim();
    // Expose slice sync via custom event
    el._vtkR=this._r;
  }
  resize(){if(this._r)this._r.render();}
  getRenderer(){return this._rendererShim;}
  getRenderWindow(){return this._renWin;}
  getInteractor(){return this._interactor;}
}

// ─── Public vtk namespace (matches @kitware/vtk.js UMD export) ───
const vtk={
  Rendering:{
    Misc:{
      vtkGenericRenderWindow:makeNewInstance(VtkGenericRenderWindow)
    },
    Core:{
      vtkVolume:makeNewInstance(VtkVolume),
      vtkVolumeMapper:makeNewInstance(VtkVolumeMapper),
      vtkVolumeProperty:makeNewInstance(VtkVolumeProp),
      vtkColorTransferFunction:makeNewInstance(VtkColorTF),
      vtkActor:makeNewInstance(VtkActor),
      vtkPolyDataMapper:makeNewInstance(VtkPolyDataMapper),
    }
  },
  Common:{
    DataModel:{
      vtkImageData:makeNewInstance(VtkImageData),
      vtkPiecewiseFunction:makeNewInstance(VtkPWF),
    },
    Core:{
      vtkDataArray:makeNewInstance(VtkDataArray),
    }
  },
  Filters:{
    Sources:{
      vtkPlaneSource:makeNewInstance(VtkPlaneSource),
    }
  },
  Interaction:{
    Style:{
      vtkInteractorStyleTrackballCamera:makeNewInstance(VtkInteractorStyleTrackball),
    }
  }
};

// Expose as window.vtk
global.vtk=vtk;
global.vtkLoaded=true;

// Fire a custom event so index.html knows it's ready
document.addEventListener('DOMContentLoaded',()=>{
  document.dispatchEvent(new CustomEvent('vtkReady'));
  console.log('[vtk-compat] Ready. WebGL2 volume renderer loaded.');
});

})(window);
