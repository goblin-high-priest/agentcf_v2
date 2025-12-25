from agentverse.registry import Registry

# ==================== 测试 Registry ====================
class TestRegistry:
    """测试注册表系统"""
    
    def test_registry_creation(self):
        """测试创建注册表"""
        registry = Registry(name="TestRegistry")
        assert registry.name == "TestRegistry"
        assert registry.entries == {}
    
    def test_register_decorator(self):
        """测试注册装饰器"""
        registry = Registry(name="TestRegistry")
        
        @registry.register("test_class")
        class TestClass:
            def __init__(self, value):
                self.value = value
        
        assert "test_class" in registry.entries
        assert registry.entries["test_class"] == TestClass
    
    def test_build_registered_class(self):
        """测试构建已注册的类"""
        registry = Registry(name="TestRegistry")
        
        @registry.register("test_class")
        class TestClass:
            def __init__(self, value):
                self.value = value
        
        instance = registry.build("test_class", value=42)
        assert instance.value == 42
    
    def test_build_unregistered_class(self):
        """测试构建未注册的类应该抛出错误"""
        registry = Registry(name="TestRegistry")
        
        with pytest.raises(ValueError) as exc_info:
            registry.build("non_existent")
        
        assert "not registered" in str(exc_info.value)
    
    def test_get_all_entries(self):
        """测试获取所有条目"""
        registry = Registry(name="TestRegistry")
        
        @registry.register("class1")
        class Class1:
            pass
        
        @registry.register("class2")
        class Class2:
            pass
        
        entries = registry.get_all_entries()
        assert len(entries) == 2
        assert "class1" in entries
        assert "class2" in entries
