#include <jni.h>
#include <pthread.h>
#include <unistd.h>
#include <string>
#include <vector>

#include <android/log.h>
#include <android/native_window.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>

#include "ImGui/imgui.h"
#include "ImGui/backends/imgui_impl_android.h"
#include "ImGui/backends/imgui_impl_vulkan.h"

#include "../Dobby/dobby.h"

#define LOG_TAG "stuffmods"
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define MAX_FRAMES_IN_FLIGHT 3

// ── Vulkan state (unchanged from base) ───────────────────────────────────────
ANativeWindow* g_NativeWindow = nullptr;
VkInstance g_Instance = VK_NULL_HANDLE;
VkPhysicalDevice g_PhysicalDevice = VK_NULL_HANDLE;
VkDevice g_Device = VK_NULL_HANDLE;
VkQueue g_Queue = VK_NULL_HANDLE;
VkDescriptorPool g_DescriptorPool = VK_NULL_HANDLE;
VkCommandPool g_CommandPool = VK_NULL_HANDLE;
VkRenderPass g_RenderPass = VK_NULL_HANDLE;
VkSurfaceKHR g_Surface = VK_NULL_HANDLE;
VkSwapchainKHR g_Swapchain = VK_NULL_HANDLE;
VkExtent2D g_SwapChainExtent = {0, 0};
std::vector<VkImage> g_SwapChainImages;
std::vector<VkFramebuffer> g_Framebuffers;
std::vector<VkImageView> g_SwapChainImageViews;

bool g_ImGuiInitialized = false;
bool g_InitInProgress = false;
static bool g_InSubmit = false;
bool g_MenuVisible = false; // toggled by left thumbstick click

struct RenderContext {
    VkCommandBuffer commandBuffer;
    VkFence fence;
    bool inUse;
    RenderContext() : commandBuffer(VK_NULL_HANDLE), fence(VK_NULL_HANDLE), inUse(false) {}
};

static RenderContext g_RenderContexts[2];
static uint32_t g_CurrentContext = 0;

// ── Game mod state ────────────────────────────────────────────────────────────
static void* g_base = nullptr;
template<typename T> static T    rp(void* p,size_t o){return *(T*)((uint8_t*)p+o);}
template<typename T> static void wp(void* p,size_t o,T v){*(T*)((uint8_t*)p+o)=v;}

static bool g_infAmmo   = false;
static bool g_rapidFire = false;
static bool g_maxCurr   = false;
static bool g_godMode   = false;
static int  g_frame     = 0;

// ── Weapon hooks ──────────────────────────────────────────────────────────────
typedef void(*ShamUpd_t)(void*); static ShamUpd_t _shamOrig=nullptr;
static void shamHook(void* s){
    if(_shamOrig)_shamOrig(s); if(!s)return;
    if(g_infAmmo){int32_t m=rp<int32_t>(s,0x88);if(m>0&&m<9999){wp<int32_t>(s,0x98,m);wp<uint8_t>(s,0x9C,0);}}
    if(g_rapidFire){wp<float>(s,0x8C,0.f);wp<uint8_t>(s,0x9C,0);}
}

typedef void(*ShotUpd_t)(void*); static ShotUpd_t _shotOrig=nullptr;
static void shotHook(void* s){
    if(_shotOrig)_shotOrig(s); if(!s)return;
    if(g_infAmmo||g_rapidFire){wp<uint8_t>(s,0x40,1);wp<uint8_t>(s,0x98,0);wp<int32_t>(s,0x9C,0);}
}

typedef void(*CGMLate_t)(void*); static CGMLate_t _cgmOrig=nullptr;
static void cgmHook(void* self){
    if(_cgmOrig)_cgmOrig(self);
    g_frame++;
    if(g_maxCurr && g_frame%300==0){
        wp<int32_t>(self,0xD0,999999);
        wp<int32_t>(self,0xE4,999999);
    }
    if(g_godMode){
        try {
            void* dpmPtr = rp<void*>(self,0x28); // GTPlayer.dethPlayerManager
            // actually get player first via CGM->selfClientPlayer->playerLocomotion
            void* scp = rp<void*>(self,0xF0); // selfClientPlayer
            if(scp){
                void* plr = rp<void*>(scp,0xB8); // playerLocomotion
                if(plr){
                    void* dpm = rp<void*>(plr,0x28); // dethPlayerManager
                    if(dpm){
                        float maxHp = rp<float>(dpm,0x98);
                        wp<float>(dpm,0xAC, maxHp>0?maxHp:100.f);
                        wp<uint8_t>(dpm,0xA8,0);
                    }
                }
            }
        } catch(...) {}
    }
}

// ── Vulkan helpers (unchanged from base) ─────────────────────────────────────
uint32_t findGraphicsQueueFamily() {
    uint32_t count=0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice,&count,nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice,&count,families.data());
    for(uint32_t i=0;i<count;i++)
        if(families[i].queueFlags&VK_QUEUE_GRAPHICS_BIT) return i;
    return UINT32_MAX;
}

VkCommandPool createCommandPool(){
    VkCommandPoolCreateInfo info{};
    info.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex=findGraphicsQueueFamily();
    VkCommandPool pool=VK_NULL_HANDLE;
    vkCreateCommandPool(g_Device,&info,nullptr,&pool);
    return pool;
}

VkCommandBuffer createCommandBuffer(){
    if(g_Device==VK_NULL_HANDLE||g_CommandPool==VK_NULL_HANDLE) return VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo info{};
    info.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandPool=g_CommandPool;
    info.commandBufferCount=1;
    VkCommandBuffer buf;
    if(vkAllocateCommandBuffers(g_Device,&info,&buf)!=VK_SUCCESS) return VK_NULL_HANDLE;
    return buf;
}

VkRenderPass createImGuiRenderPass(){
    VkAttachmentDescription att={};
    att.format=VK_FORMAT_B8G8R8A8_UNORM;
    att.samples=VK_SAMPLE_COUNT_1_BIT;
    att.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD;
    att.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    att.finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference ref={0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub={};
    sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount=1;
    sub.pColorAttachments=&ref;
    VkSubpassDependency dep={};
    dep.srcSubpass=VK_SUBPASS_EXTERNAL;
    dep.dstSubpass=0;
    dep.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    dep.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo info={};
    info.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount=1;
    info.pAttachments=&att;
    info.subpassCount=1;
    info.pSubpasses=&sub;
    info.dependencyCount=1;
    info.pDependencies=&dep;
    VkRenderPass rp;
    if(vkCreateRenderPass(g_Device,&info,nullptr,&rp)!=VK_SUCCESS) return VK_NULL_HANDLE;
    return rp;
}

void createImGuiFramebuffers(){
    g_Framebuffers.resize(g_SwapChainImages.size());
    g_SwapChainImageViews.resize(g_SwapChainImages.size());
    for(size_t i=0;i<g_SwapChainImages.size();i++){
        VkImageViewCreateInfo ci={};
        ci.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image=g_SwapChainImages[i];
        ci.viewType=VK_IMAGE_VIEW_TYPE_2D;
        ci.format=VK_FORMAT_B8G8R8A8_UNORM;
        ci.components={VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,
                       VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY};
        ci.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        if(vkCreateImageView(g_Device,&ci,nullptr,&g_SwapChainImageViews[i])!=VK_SUCCESS) continue;
        VkImageView atts[]={g_SwapChainImageViews[i]};
        VkFramebufferCreateInfo fi={};
        fi.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass=g_RenderPass;
        fi.attachmentCount=1;
        fi.pAttachments=atts;
        fi.width=g_SwapChainExtent.width;
        fi.height=g_SwapChainExtent.height;
        fi.layers=1;
        vkCreateFramebuffer(g_Device,&fi,nullptr,&g_Framebuffers[i]);
    }
}

bool createDescriptorPool(){
    VkDescriptorPoolSize sizes[]={
        {VK_DESCRIPTOR_TYPE_SAMPLER,1000},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1000},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1000},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,1000},{VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1000},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1000},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,1000},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,1000}
    };
    VkDescriptorPoolCreateInfo info={};
    info.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    info.maxSets=1000;
    info.poolSizeCount=std::size(sizes);
    info.pPoolSizes=sizes;
    return vkCreateDescriptorPool(g_Device,&info,nullptr,&g_DescriptorPool)==VK_SUCCESS;
}

void initRenderContexts(){
    for(int i=0;i<2;i++){
        g_RenderContexts[i].commandBuffer=createCommandBuffer();
        VkFenceCreateInfo fi={};
        fi.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fi.flags=VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(g_Device,&fi,nullptr,&g_RenderContexts[i].fence);
        g_RenderContexts[i].inUse=false;
    }
}

bool uploadFonts(){
    VkCommandBuffer cb=createCommandBuffer();
    if(!cb) return false;
    VkCommandBufferBeginInfo bi={};
    bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if(vkBeginCommandBuffer(cb,&bi)!=VK_SUCCESS){vkFreeCommandBuffers(g_Device,g_CommandPool,1,&cb);return false;}
    ImGui_ImplVulkan_CreateFontsTexture();
    if(vkEndCommandBuffer(cb)!=VK_SUCCESS){vkFreeCommandBuffers(g_Device,g_CommandPool,1,&cb);return false;}
    VkSubmitInfo si={};
    si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount=1;
    si.pCommandBuffers=&cb;
    VkFence fence;
    VkFenceCreateInfo fi={};
    fi.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if(vkCreateFence(g_Device,&fi,nullptr,&fence)!=VK_SUCCESS){vkFreeCommandBuffers(g_Device,g_CommandPool,1,&cb);return false;}
    vkQueueSubmit(g_Queue,1,&si,fence);
    vkWaitForFences(g_Device,1,&fence,VK_TRUE,UINT64_MAX);
    vkDestroyFence(g_Device,fence,nullptr);
    vkFreeCommandBuffers(g_Device,g_CommandPool,1,&cb);
    return true;
}

bool initializeImGui(){
    if(!createDescriptorPool()) return false;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io=ImGui::GetIO();
    io.IniFilename=NULL;
    io.ConfigFlags|=ImGuiConfigFlags_NoMouseCursorChange;
    io.DisplaySize=ImVec2(g_SwapChainExtent.width,g_SwapChainExtent.height);

    // Dark theme with orange accent - matches the screenshot style
    ImGui::StyleColorsDark();
    ImGuiStyle& s=ImGui::GetStyle();
    s.WindowRounding=4.f;
    s.FrameRounding=3.f;
    s.WindowBorderSize=1.f;
    s.WindowTitleAlign=ImVec2(0.5f,0.5f);
    ImVec4* c=s.Colors;
    c[ImGuiCol_WindowBg]       =ImVec4(0.06f,0.06f,0.08f,0.92f);
    c[ImGuiCol_TitleBg]        =ImVec4(0.10f,0.05f,0.00f,1.f);
    c[ImGuiCol_TitleBgActive]  =ImVec4(0.20f,0.10f,0.00f,1.f);
    c[ImGuiCol_Button]         =ImVec4(0.20f,0.10f,0.00f,1.f);
    c[ImGuiCol_ButtonHovered]  =ImVec4(0.80f,0.40f,0.00f,1.f);
    c[ImGuiCol_ButtonActive]   =ImVec4(1.00f,0.55f,0.00f,1.f);
    c[ImGuiCol_CheckMark]      =ImVec4(1.00f,0.55f,0.00f,1.f);
    c[ImGuiCol_FrameBg]        =ImVec4(0.15f,0.08f,0.00f,1.f);
    c[ImGuiCol_Header]         =ImVec4(0.20f,0.10f,0.00f,1.f);
    c[ImGuiCol_HeaderHovered]  =ImVec4(0.80f,0.40f,0.00f,1.f);
    c[ImGuiCol_Separator]      =ImVec4(1.00f,0.55f,0.00f,0.5f);

    if(!ImGui_ImplAndroid_Init(g_NativeWindow)) return false;
    g_RenderPass=createImGuiRenderPass();
    if(!g_RenderPass) return false;
    createImGuiFramebuffers();
    if(g_Framebuffers.empty()) return false;

    ImGui_ImplVulkan_InitInfo ii={};
    ii.Instance=g_Instance;
    ii.PhysicalDevice=g_PhysicalDevice;
    ii.Device=g_Device;
    ii.QueueFamily=findGraphicsQueueFamily();
    ii.Queue=g_Queue;
    ii.RenderPass=g_RenderPass;
    ii.DescriptorPool=g_DescriptorPool;
    ii.MinImageCount=2;
    ii.ImageCount=g_SwapChainImages.size();
    ii.MSAASamples=VK_SAMPLE_COUNT_1_BIT;
    if(!ImGui_ImplVulkan_Init(&ii)) return false;
    return uploadFonts();
}

// ── Menu UI ───────────────────────────────────────────────────────────────────
static void Checkbox(const char* label, bool* val){
    ImGui::Checkbox(label, val);
}

void DrawMenu(){
    ImGui::SetNextWindowPos(ImVec2(10,10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(240,320), ImGuiCond_FirstUseEver);
    ImGui::Begin("StuffMods", nullptr,
        ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoResize);

    ImGui::TextColored(ImVec4(1,0.55f,0,1), "=== Weapons ===");
    Checkbox("Inf Ammo",   &g_infAmmo);
    Checkbox("Rapid Fire", &g_rapidFire);
    ImGui::Separator();

    ImGui::TextColored(ImVec4(1,0.55f,0,1), "=== Player ===");
    Checkbox("God Mode",   &g_godMode);
    ImGui::Separator();

    ImGui::TextColored(ImVec4(1,0.55f,0,1), "=== Currency ===");
    Checkbox("Max Currency", &g_maxCurr);
    ImGui::Separator();

    ImGui::TextColored(ImVec4(0.5f,0.5f,0.5f,1),
        "Frame: %d", g_frame);
    ImGui::End();
}

// ── vkQueueSubmit hook (from base, unchanged) ─────────────────────────────────
VkResult (*original_vkQueueSubmit)(VkQueue,uint32_t,const VkSubmitInfo*,VkFence);
VkResult hooked_vkQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence){
    if(g_InSubmit) return original_vkQueueSubmit(queue,submitCount,pSubmits,fence);
    g_InSubmit=true;
    VkResult result=original_vkQueueSubmit(queue,submitCount,pSubmits,fence);
    if(result!=VK_SUCCESS||!g_ImGuiInitialized){g_InSubmit=false;return result;}

    vkQueueWaitIdle(queue);

    try {
        RenderContext& ctx=g_RenderContexts[g_CurrentContext];
        if(!ctx.commandBuffer){g_InSubmit=false;return result;}
        if(ctx.inUse){
            if(vkWaitForFences(g_Device,1,&ctx.fence,VK_TRUE,1000000)!=VK_SUCCESS){g_InSubmit=false;return result;}
            vkResetFences(g_Device,1,&ctx.fence);
        }
        vkResetCommandBuffer(ctx.commandBuffer,0);
        VkCommandBufferBeginInfo bi={};
        bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if(vkBeginCommandBuffer(ctx.commandBuffer,&bi)!=VK_SUCCESS){g_InSubmit=false;return result;}

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplAndroid_NewFrame();
        ImGui::NewFrame();
        DrawMenu();
        ImGui::Render();

        ImDrawData* dd=ImGui::GetDrawData();
        if(dd){
            uint32_t idx=0;
            if(pSubmits&&pSubmits->commandBufferCount>0)
                idx=reinterpret_cast<uintptr_t>(pSubmits->pCommandBuffers[0])%g_Framebuffers.size();
            VkRenderPassBeginInfo rpi={};
            rpi.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rpi.renderPass=g_RenderPass;
            rpi.framebuffer=g_Framebuffers[idx];
            rpi.renderArea={{0,0},g_SwapChainExtent};
            vkCmdBeginRenderPass(ctx.commandBuffer,&rpi,VK_SUBPASS_CONTENTS_INLINE);
            ImGui_ImplVulkan_RenderDrawData(dd,ctx.commandBuffer);
            vkCmdEndRenderPass(ctx.commandBuffer);
        }
        if(vkEndCommandBuffer(ctx.commandBuffer)!=VK_SUCCESS){g_InSubmit=false;return result;}

        VkSubmitInfo si={};
        si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount=1;
        si.pCommandBuffers=&ctx.commandBuffer;
        vkQueueSubmit(queue,1,&si,ctx.fence);
        ctx.inUse=true;
        g_CurrentContext=(g_CurrentContext+1)%2;
    } catch(...) {}

    g_InSubmit=false;
    return result;
}

// ── Vulkan creation hooks (from base, unchanged) ──────────────────────────────
VkResult (*vkCreateInstanceOrigin)(const VkInstanceCreateInfo*,const VkAllocationCallbacks*,VkInstance*);
VkResult vkCreateInstanceReplace(const VkInstanceCreateInfo* pCI,const VkAllocationCallbacks* pA,VkInstance* pI){
    VkResult r=vkCreateInstanceOrigin(pCI,pA,pI);
    if(r==VK_SUCCESS){g_Instance=*pI;LOGD("vkCreateInstance");}
    return r;
}

VkResult (*vkCreateDeviceOrigin)(VkPhysicalDevice,const VkDeviceCreateInfo*,const VkAllocationCallbacks*,VkDevice*);
VkResult vkCreateDeviceReplace(VkPhysicalDevice pd,const VkDeviceCreateInfo* pCI,const VkAllocationCallbacks* pA,VkDevice* pD){
    VkResult r=vkCreateDeviceOrigin(pd,pCI,pA,pD);
    if(r==VK_SUCCESS){
        g_PhysicalDevice=pd; g_Device=*pD;
        vkGetDeviceQueue(g_Device,findGraphicsQueueFamily(),0,&g_Queue);
        g_CommandPool=createCommandPool();
        LOGD("Device created");
    }
    return r;
}

VkResult (*vkCreateAndroidSurfaceKHROrigin)(VkInstance,const VkAndroidSurfaceCreateInfoKHR*,const VkAllocationCallbacks*,VkSurfaceKHR*);
VkResult vkCreateAndroidSurfaceKHRReplace(VkInstance inst,const VkAndroidSurfaceCreateInfoKHR* pCI,const VkAllocationCallbacks* pA,VkSurfaceKHR* pS){
    if(pCI&&!g_NativeWindow){g_NativeWindow=pCI->window;LOGD("NativeWindow captured");}
    return vkCreateAndroidSurfaceKHROrigin(inst,pCI,pA,pS);
}

VkResult (*vkCreateSwapchainKHROrigin)(VkDevice,const VkSwapchainCreateInfoKHR*,const VkAllocationCallbacks*,VkSwapchainKHR*);
VkResult vkCreateSwapchainKHRReplace(VkDevice dev,const VkSwapchainCreateInfoKHR* pCI,const VkAllocationCallbacks* pA,VkSwapchainKHR* pSC){
    VkResult r=vkCreateSwapchainKHROrigin(dev,pCI,pA,pSC);
    if(r==VK_SUCCESS){
        g_Swapchain=*pSC; g_SwapChainExtent=pCI->imageExtent; g_Surface=pCI->surface;
        uint32_t cnt;
        vkGetSwapchainImagesKHR(dev,g_Swapchain,&cnt,nullptr);
        g_SwapChainImages.resize(cnt);
        vkGetSwapchainImagesKHR(dev,g_Swapchain,&cnt,g_SwapChainImages.data());
        LOGD("Swapchain %dx%d images=%d",g_SwapChainExtent.width,g_SwapChainExtent.height,cnt);
        if(!g_ImGuiInitialized&&!g_InitInProgress&&g_Device!=VK_NULL_HANDLE&&g_Queue!=VK_NULL_HANDLE&&g_NativeWindow){
            g_InitInProgress=true;
            if(initializeImGui()){
                initRenderContexts();
                g_ImGuiInitialized=true;
                LOGD("ImGui initialized!");
            } else LOGD("ImGui init FAILED");
            g_InitInProgress=false;
        }
    }
    return r;
}

// ── Touch / input ─────────────────────────────────────────────────────────────
void (*initializeMotionEventOrigin)(void*,void*,void*);
void initializeMotionEventReplace(void* thiz,void* event,void* msg){
    initializeMotionEventOrigin(thiz,event,msg);
    ImGui_ImplAndroid_HandleInputEvent((AInputEvent*)thiz);
}

// Left thumbstick click toggle
// dispatchKeyEvent in InputConsumer - keycode 106 = AKEYCODE_BUTTON_THUMBL
// Or hook via the key dispatch path
#include <android/input.h>

// We check OVR button state directly via a simple polling approach
// AKEYCODE_BUTTON_THUMBL = 106, AKEYCODE_BUTTON_THUMBR = 107
// These arrive as key events through the same libinput path
// Hook dispatchKeyEvent to catch thumbstick presses
void (*dispatchKeyEventOrigin)(void*, void*, void*);
void dispatchKeyEventReplace(void* thiz, void* event, void* msg){
    if(dispatchKeyEventOrigin) dispatchKeyEventOrigin(thiz, event, msg);
    // AInputEvent key code is at offset 0x14 in InputMessage for key events
    // Simpler: just check the raw event action/keycode
    AInputEvent* ke = (AInputEvent*)thiz;
    if(!ke) return;
    int type = AInputEvent_getType(ke);
    if(type == AINPUT_EVENT_TYPE_KEY){
        int action  = AKeyEvent_getAction(ke);
        int keycode = AKeyEvent_getKeyCode(ke);
        // AKEYCODE_BUTTON_THUMBL = 106
        if(keycode == 106 && action == AKEY_EVENT_ACTION_DOWN){
            g_MenuVisible = !g_MenuVisible;
            LOGD("thumbstick toggle: menu=%d", g_MenuVisible);
        }
    }
}

bool isLibraryLoaded(const char* name){
    FILE* fp=fopen("/proc/self/maps","rt"); char line[512]={};
    if(fp){while(fgets(line,sizeof(line),fp))if(strstr(line,name)){fclose(fp);return true;}fclose(fp);}
    return false;
}

// ── Game hook setup ───────────────────────────────────────────────────────────
void setupGameHooks(){
    // Find libil2cpp.so base
    FILE* m=fopen("/proc/self/maps","r"); char line[512];
    while(m&&fgets(line,sizeof(line),m)){
        if(strstr(line,"libil2cpp.so")&&strstr(line,"r-xp")){
            uint64_t b=0; sscanf(line,"%llx",(unsigned long long*)&b);
            g_base=(void*)b; LOGD("il2cpp base=%p",g_base); break;
        }
    }
    if(m)fclose(m);
    if(!g_base){LOGD("il2cpp not found");return;}

    auto rva=[](uint64_t o)->void*{return (uint8_t*)g_base+o;};
    DobbyHook(rva(0x20CA130),(void*)shamHook,(void**)&_shamOrig);
    DobbyHook(rva(0x20E0304),(void*)shotHook,(void**)&_shotOrig);
    DobbyHook(rva(0x20AFB28),(void*)cgmHook, (void**)&_cgmOrig);
    LOGD("Game hooks installed");
}

void initializeHooks(){
    do { usleep(100); } while(!isLibraryLoaded("libvulkan.so"));

    DobbyHook(DobbySymbolResolver("libvulkan.so","vkCreateInstance"),
        (void*)vkCreateInstanceReplace,(void**)&vkCreateInstanceOrigin);
    DobbyHook(DobbySymbolResolver("libvulkan.so","vkCreateDevice"),
        (void*)vkCreateDeviceReplace,(void**)&vkCreateDeviceOrigin);
    DobbyHook(DobbySymbolResolver("libvulkan.so","vkCreateAndroidSurfaceKHR"),
        (void*)vkCreateAndroidSurfaceKHRReplace,(void**)&vkCreateAndroidSurfaceKHROrigin);
    DobbyHook(DobbySymbolResolver("libvulkan.so","vkCreateSwapchainKHR"),
        (void*)vkCreateSwapchainKHRReplace,(void**)&vkCreateSwapchainKHROrigin);
    DobbyHook(DobbySymbolResolver("libvulkan.so","vkQueueSubmit"),
        (void*)hooked_vkQueueSubmit,(void**)&original_vkQueueSubmit);

    auto inputAddr=DobbySymbolResolver("libinput.so",
        "_ZN7android13InputConsumer21initializeMotionEventEPNS_11MotionEventEPKNS_12InputMessageE");
    if(inputAddr) DobbyHook(inputAddr,(void*)initializeMotionEventReplace,(void**)&initializeMotionEventOrigin);

    // Hook key events for thumbstick toggle
    auto keyEventAddr = DobbySymbolResolver("libinput.so",
        "_ZN7android13InputConsumer19dispatchKeyEventEjPNS_8KeyEventE");
    if(keyEventAddr){
        DobbyHook(keyEventAddr,(void*)dispatchKeyEventReplace,(void**)&dispatchKeyEventOrigin);
        LOGD("Key event hook installed");
    } else {
        LOGD("Key event hook not found - thumbstick toggle unavailable");
    }

    LOGD("Vulkan hooks installed");

    // Wait a bit then hook game methods
    usleep(5000000); // 5s
    setupGameHooks();
}

void* menuThread(void*){ initializeHooks(); return nullptr; }

__attribute__((constructor))
void init(){
    LOGD("stuffmods loaded!");
    pthread_t t;
    pthread_create(&t,nullptr,menuThread,nullptr);
}
